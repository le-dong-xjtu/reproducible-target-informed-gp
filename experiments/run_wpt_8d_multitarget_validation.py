import csv
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_automatic_gp_wpt8d as auto

TARGETS = tuple(float(value) for value in range(1000, 2001, 50))
POLICIES = (
    "automated",
    "center_pf_cei_matched",
    "conventional_cei_matched",
    "sobol_matched",
    "forced_b50",
    "conventional_cei_b50",
)
OUT = (
    auto.PROJECT
    / "results"
    / "validation"
    / "wpt_8d_multitarget_validation_uncapped"
)


def components(models, candidates, observed, target):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_m, sd_m = models[0].predict(candidates, return_std=True)
        mu_r, sd_r = models[1].predict(candidates, return_std=True)
    if not all(np.isfinite(v).all() for v in (mu_m, sd_m, mu_r, sd_r)):
        raise FloatingPointError("non-finite multi-target GP prediction")
    sd_m = np.maximum(sd_m, 1e-12)
    sd_r = np.maximum(sd_r, 1e-12)
    lower, upper = target * (1 - auto.EPSILON), target * (1 + auto.EPSILON)
    pf = ndtr((upper - mu_m) / sd_m) - ndtr((lower - mu_m) / sd_m)
    feasible = observed[np.abs(observed[:, 0] - target) <= auto.EPSILON * target]
    if len(feasible):
        incumbent = float(feasible[:, 1].min())
        improvement = incumbent - mu_r
        z = improvement / sd_r
        ei = (
            improvement * ndtr(z)
            + sd_r * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        )
        cei = pf * np.maximum(ei, 0) / incumbent
    else:
        incumbent, cei = None, None
    return pf, mu_r, sd_m, cei, incumbent


def oracle(point, timing):
    started = time.perf_counter()
    value = auto.evaluate(point)
    timing["oracle"] += time.perf_counter() - started
    return value


def confirmation(x_raw, y, target, stream, timing):
    started = time.perf_counter()
    begin = auto.CONFIRM_POOL_START
    candidates = auto.unseen_unique(
        stream[begin : begin + auto.N_CONFIRM_CANDIDATES], x_raw
    )
    models = auto.gps(auto.normalize(x_raw), y)
    pf, mu_r, _, _, _ = components(models, auto.normalize(candidates), y, target)
    score = pf * np.exp(-auto.unit(mu_r))
    chosen = np.argsort(-score, kind="stable")[: auto.N_CONFIRM]
    timing["surrogate"] += time.perf_counter() - started
    x_confirm = candidates[chosen]
    y_confirm = np.asarray([oracle(point, timing) for point in x_confirm])
    return x_confirm, y_confirm


def finish(
    policy,
    target,
    x_model,
    y_model,
    x_confirm,
    y_confirm,
    timing,
    first_feasible,
    stopped,
    stop_rho,
):
    x_all, y_all = np.vstack([x_model, x_confirm]), np.vstack([y_model, y_confirm])
    feasible = np.where(np.abs(y_all[:, 0] - target) <= auto.EPSILON * target)[0]
    qualified = bool(len(feasible))
    decision = (
        int(feasible[np.argmin(y_all[feasible, 1])])
        if qualified
        else int(np.argmin(np.abs(y_all[:, 0] - target)))
    )
    return {
        "policy": policy,
        "target_M_nH": target,
        "qualified": qualified,
        "best_M_nH": float(y_all[decision, 0]),
        "best_R_ohm": float(y_all[decision, 1]),
        "relative_target_error_pct": float(
            abs(y_all[decision, 0] - target) / target * 100
        ),
        "feasibility_margin_nH": float(
            auto.EPSILON * target - abs(y_all[decision, 0] - target)
        ),
        "design": json.dumps(x_all[decision].tolist()),
        "decision_stage": (
            "model_building" if decision < len(x_model) else "confirmation"
        ),
        "first_feasible_evaluation": first_feasible or "",
        "n_model_building": len(x_model),
        "n_confirmation": len(x_confirm),
        "total_evaluations": len(x_all),
        "stopped_naturally": stopped,
        "stop_rho": "" if stop_rho is None else stop_rho,
        "oracle_runtime_sec": timing["oracle"],
        "surrogate_runtime_sec": timing["surrogate"],
        "total_runtime_sec": timing["oracle"] + timing["surrogate"],
    }


def run_gp(
    policy,
    target,
    force,
    conventional,
    model_cap=auto.MAX_MODEL_CALLS,
    center_initial=False,
):
    timing = {"oracle": 0.0, "surrogate": 0.0}
    stream = auto.sobol_stream()
    if conventional and not center_initial:
        x_raw = auto.unseen_unique(stream[1 : 1 + 16], np.empty((0, 8)))[:6]
    else:
        x_raw = np.asarray([auto.CENTER], float)
    y = np.asarray([oracle(point, timing) for point in x_raw])
    feasible_mask = np.abs(y[:, 0] - target) <= auto.EPSILON * target
    first_feasible = (
        int(np.flatnonzero(feasible_mask)[0] + 1) if feasible_mask.any() else None
    )
    streak, stop_rho, stopped = 0, None, False
    while (not force and not conventional) or len(x_raw) < model_cap:
        step = len(x_raw) - 1
        block = stream[
            1 + step * auto.N_CANDIDATES : 1 + (step + 1) * auto.N_CANDIDATES
        ]
        candidates = auto.unseen_unique(block, x_raw)
        started = time.perf_counter()
        models = auto.gps(auto.normalize(x_raw), y)
        pf, _, sd_m, cei, _ = components(
            models, auto.normalize(candidates), y, target
        )
        if first_feasible is None:
            score = (
                pf
                if conventional
                else auto.FEASIBILITY_WEIGHT * auto.unit(pf)
                + (1 - auto.FEASIBILITY_WEIGHT) * auto.unit(sd_m)
            )
        else:
            stop_rho = float(np.max(cei))
            if not force and not conventional:
                streak = streak + 1 if stop_rho <= auto.STOP_DELTA else 0
                if (
                    len(x_raw) >= auto.MIN_MODEL_CALLS
                    and streak >= auto.STOP_PATIENCE
                ):
                    timing["surrogate"] += time.perf_counter() - started
                    stopped = True
                    break
            score = cei
        chosen = candidates[int(np.argmax(score))]
        timing["surrogate"] += time.perf_counter() - started
        value = oracle(chosen, timing)
        x_raw, y = np.vstack([x_raw, chosen]), np.vstack([y, value])
        if first_feasible is None and abs(value[0] - target) <= auto.EPSILON * target:
            first_feasible = len(x_raw)
    x_confirm, y_confirm = confirmation(x_raw, y, target, stream, timing)
    return finish(
        policy,
        target,
        x_raw,
        y,
        x_confirm,
        y_confirm,
        timing,
        first_feasible,
        stopped,
        stop_rho,
    )


def run_sobol(target, budget):
    timing = {"oracle": 0.0, "surrogate": 0.0}
    unique = []
    seen = set()
    for point in auto.canonical(auto.sobol_stream()[1:]):
        key = tuple(point)
        if key not in seen:
            unique.append(point)
            seen.add(key)
        if len(unique) == budget:
            break
    x = np.asarray(unique)
    y = np.asarray([oracle(point, timing) for point in x])
    feasible = np.abs(y[:, 0] - target) <= auto.EPSILON * target
    first = int(np.flatnonzero(feasible)[0] + 1) if feasible.any() else None
    return finish(
        "sobol_matched",
        target,
        x,
        y,
        np.empty((0, 8)),
        np.empty((0, 2)),
        timing,
        first,
        False,
        None,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in TARGETS:
        automated = run_gp("automated", target, force=False, conventional=False)
        rows.append(automated)
        matched_cap = automated["n_model_building"]
        rows.append(
            run_gp(
                "center_pf_cei_matched",
                target,
                force=True,
                conventional=True,
                model_cap=matched_cap,
                center_initial=True,
            )
        )
        rows.append(
            run_gp(
                "conventional_cei_matched",
                target,
                force=True,
                conventional=True,
                model_cap=matched_cap,
            )
        )
        rows.append(run_gp("forced_b50", target, force=True, conventional=False))
        rows.append(
            run_gp("conventional_cei_b50", target, force=True, conventional=True)
        )
        rows.append(run_sobol(target, automated["total_evaluations"]))
        print("completed target", target, flush=True)
    with (OUT / "results.csv").open("w", newline="") as stream_out:
        writer = csv.DictWriter(stream_out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for policy in POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        qualified = [row for row in group if row["qualified"]]
        summary.append(
            {
                "policy": policy,
                "cases": len(group),
                "qualified_cases": len(qualified),
                "mean_R_qualified_ohm": (
                    float(np.mean([r["best_R_ohm"] for r in qualified]))
                    if qualified
                    else None
                ),
                "mean_target_error_pct": float(
                    np.mean([r["relative_target_error_pct"] for r in group])
                ),
                "mean_total_evaluations": float(
                    np.mean([r["total_evaluations"] for r in group])
                ),
                "mean_oracle_runtime_sec": float(
                    np.mean([r["oracle_runtime_sec"] for r in group])
                ),
                "mean_surrogate_runtime_sec": float(
                    np.mean([r["surrogate_runtime_sec"] for r in group])
                ),
            }
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "targets_nH": list(TARGETS),
                "relative_tolerance": auto.EPSILON,
                "policies": list(POLICIES),
                "deterministic": True,
                "shared_gp": {
                    "kernel": "Constant * isotropic Matern-5/2 + White",
                    "constant_initial": 1.0,
                    "constant_bounds": [1e-3, 1e3],
                    "length_scale_initial": 0.35,
                    "length_scale_bounds": [0.03, 3.0],
                    "white_initial": 1e-5,
                    "white_bounds": [1e-8, 1e-2],
                    "alpha": 1e-6,
                    "normalize_y": True,
                    "optimizer": "deterministic Halton multistart L-BFGS-B",
                    "starts": 5,
                },
                "shared_candidates": "unscrambled Sobol blocks",
                "automated_parameters_frozen": {
                    "omega": auto.FEASIBILITY_WEIGHT,
                    "delta": auto.STOP_DELTA,
                    "patience": auto.STOP_PATIENCE,
                    "minimum_model_building": auto.MIN_MODEL_CALLS,
                    "confirmation": auto.N_CONFIRM,
                },
            },
            indent=2,
        )
        + "\n"
    )
    with (OUT / "aedt_cases.csv").open("w", newline="") as stream_out:
        fields = [
            "case_name",
            "target_M_nH",
            "partial_M_nH",
            "analytic_R_ohm",
            "ns_t",
            "nt_t",
            "s_t_mm",
            "ri_t_mm",
            "ns_r",
            "nt_r",
            "s_r_mm",
            "ri_r_mm",
            "z_gap_mm",
        ]
        writer = csv.DictWriter(stream_out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row["policy"] != "automated":
                continue
            design = json.loads(row["design"])
            writer.writerow(
                dict(
                    zip(
                        fields,
                        [
                            f"M{int(row['target_M_nH']):04d}",
                            row["target_M_nH"],
                            row["best_M_nH"],
                            row["best_R_ohm"],
                        ]
                        + design
                        + [70.0],
                    )
                )
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
