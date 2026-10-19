import argparse
import csv
import json
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import qmc, wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "essay" / "data" / "center_start_cei_lr"
ROWS = OUT / "results.csv"
SEEDS = tuple(range(20261015, 20261025))
TARGETS = tuple(range(1, 88))
EPSILON = 0.025
N_MODEL, N_CONFIRM = 17, 10

wire_diameter = 0.68
pitch = 1.05 * wire_diameter
resistivity = 1.851e-5
POINTS = [(layers, turns) for layers in range(1, 11) for turns in range(20, 131)]
X = np.asarray(
    [[(layers - 1) / 9, (turns - 20) / 110] for layers, turns in POINTS], float
)
L_ALL = np.asarray(
    [
        0.08
        * (layers * turns) ** 2
        * (80 + layers * pitch) ** 2
        * 1e-4
        / (3 * (80 + layers * pitch) + 9 * turns * pitch + 10 * layers * pitch)
        for layers, turns in POINTS
    ]
)
R_ALL = np.asarray(
    [
        4 * resistivity * layers * turns * (80 + layers * pitch) / wire_diameter**2
        for layers, turns in POINTS
    ]
)
CENTER_ID = int(np.argmin(np.sum((X - 0.5) ** 2, axis=1)))
FIELDS = (
    "method",
    "budget",
    "n_initial",
    "n_sequential",
    "n_confirmation",
    "seed",
    "target_mH",
    "qualified",
    "exact",
    "bounded_loss_u",
    "relative_resistance_regret",
    "decision_T",
    "decision_n",
    "decision_L_mH",
    "decision_R_ohm",
    "decision_source",
    "first_feasible_evaluation",
    "model_building_ids",
    "confirmation_ids",
)


def kernel():
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF([0.25, 0.25], (1e-2, 10.0))
        + WhiteKernel(1e-6, (1e-8, 1e-2))
    )


def deterministic_multistart(starts):
    if starts < 1:
        raise ValueError("deterministic starts must be positive")

    def optimizer(objective, initial_theta, bounds):
        candidates = [np.asarray(initial_theta, float)]
        if starts > 1:
            sampler = qmc.Halton(d=len(initial_theta), scramble=False)
            sampler.fast_forward(1)
            unit_points = sampler.random(starts - 1)
            candidates.extend(qmc.scale(unit_points, bounds[:, 0], bounds[:, 1]))
        best_theta, best_value = None, np.inf
        for theta in candidates:
            result = minimize(
                objective, theta, method="L-BFGS-B", jac=True, bounds=bounds
            )
            if result.fun < best_value:
                best_theta, best_value = result.x, float(result.fun)
        return best_theta, best_value

    return optimizer


def fit(values, ids, seed, final=False, deterministic_starts=0):
    optimize = len(ids) >= 4
    gp_optimizer = (
        deterministic_multistart(deterministic_starts)
        if optimize and deterministic_starts
        else "fmin_l_bfgs_b" if optimize else None
    )
    model = GaussianProcessRegressor(
        kernel=kernel(),
        alpha=1e-10,
        normalize_y=True,
        random_state=seed,
        optimizer=gp_optimizer,
        n_restarts_optimizer=(
            (5 if final else 3) if optimize and not deterministic_starts else 0
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X[ids], values[ids])
        mean, std = model.predict(X, return_std=True)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise FloatingPointError("non-finite GP prediction")
    return mean, std


def unit(values):
    values = np.asarray(values, float)
    span = np.ptp(values)
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def is_feasible(index, target):
    return abs(L_ALL[index] - target) / target <= EPSILON


def run_one(task):
    seed, target = task[:2]
    pre_uncertainty_weight = float(task[2]) if len(task) > 2 else 0.0
    if not 0 <= pre_uncertainty_weight <= 1:
        raise ValueError("pre-feasibility uncertainty weight must be in [0, 1]")
    selected = [CENTER_ID]
    selected_set = {CENTER_ID}
    first_feasible = 1 if is_feasible(CENTER_ID, target) else None

    while len(selected) < N_MODEL:
        ids = np.asarray(selected, int)
        mean_l, std_l = fit(L_ALL, ids, seed)
        mean_r, std_r = fit(R_ALL, ids, seed)
        unseen = np.asarray(
            [i for i in range(len(POINTS)) if i not in selected_set], int
        )
        sigma_l = np.maximum(std_l[unseen], 1e-12)
        lower, upper = target * (1 - EPSILON), target * (1 + EPSILON)
        p_f = ndtr((upper - mean_l[unseen]) / sigma_l) - ndtr(
            (lower - mean_l[unseen]) / sigma_l
        )
        feasible_ids = [i for i in selected if is_feasible(i, target)]
        if feasible_ids:
            incumbent = min(R_ALL[i] for i in feasible_ids)
            sigma_r = np.maximum(std_r[unseen], 1e-12)
            z = (incumbent - mean_r[unseen]) / sigma_r
            ei = (incumbent - mean_r[unseen]) * ndtr(z) + sigma_r * np.exp(
                -0.5 * z * z
            ) / np.sqrt(2 * np.pi)
            score = p_f * np.maximum(ei, 0) / incumbent
        else:
            if pre_uncertainty_weight:
                score = (1 - pre_uncertainty_weight) * unit(
                    p_f
                ) + pre_uncertainty_weight * unit(sigma_l)
            else:
                score = p_f
        chosen = int(unseen[int(np.argmax(score))])
        selected.append(chosen)
        selected_set.add(chosen)
        if first_feasible is None and is_feasible(chosen, target):
            first_feasible = len(selected)

    ids = np.asarray(selected, int)
    mean_l, std_l = fit(L_ALL, ids, seed, final=True)
    mean_r, _ = fit(R_ALL, ids, seed, final=True)
    unseen = np.asarray(
        [i for i in range(len(POINTS)) if i not in selected_set], int
    )
    sigma_l = np.maximum(std_l[unseen], 1e-12)
    lower, upper = target * (1 - EPSILON), target * (1 + EPSILON)
    p_f = ndtr((upper - mean_l[unseen]) / sigma_l) - ndtr(
        (lower - mean_l[unseen]) / sigma_l
    )
    confirmation_score = p_f * np.exp(-unit(mean_r[unseen]))
    confirmation = [
        int(i) for i in unseen[np.argsort(-confirmation_score)[:N_CONFIRM]]
    ]

    pool = selected + confirmation
    feasible_pool = [i for i in pool if is_feasible(i, target)]
    qualified = bool(feasible_pool)
    if qualified:
        decision = min(feasible_pool, key=lambda i: R_ALL[i])
        feasible_domain = np.flatnonzero(
            np.abs(L_ALL - target) / target <= EPSILON
        )
        optimum = int(feasible_domain[np.argmin(R_ALL[feasible_domain])])
        regret = float(R_ALL[decision] / R_ALL[optimum] - 1)
        loss = regret / (1 + regret)
        exact = regret <= 1e-12
    else:
        decision = min(pool, key=lambda i: abs(L_ALL[i] - target) / target)
        regret, loss, exact = None, 1.0, False
    source = (
        "center_initial"
        if decision == CENTER_ID
        else "sequential" if decision in selected_set else "confirmation"
    )
    layers, turns = POINTS[decision]
    return {
        "method": (
            "center_start_cei"
            if pre_uncertainty_weight == 0
            else f"center_feasibility_uncertainty_{pre_uncertainty_weight:g}"
        ),
        "budget": N_MODEL + N_CONFIRM,
        "n_initial": 1,
        "n_sequential": N_MODEL - 1,
        "n_confirmation": N_CONFIRM,
        "seed": seed,
        "target_mH": target,
        "qualified": qualified,
        "exact": exact,
        "bounded_loss_u": loss,
        "relative_resistance_regret": regret,
        "decision_T": layers,
        "decision_n": turns,
        "decision_L_mH": float(L_ALL[decision]),
        "decision_R_ohm": float(R_ALL[decision]),
        "decision_source": source,
        "first_feasible_evaluation": first_feasible,
        "model_building_ids": json.dumps(selected),
        "confirmation_ids": json.dumps(confirmation),
    }


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(rows):
    values = np.asarray([float(row["bounded_loss_u"]) for row in rows])
    by_target, by_seed = defaultdict(list), defaultdict(list)
    for row in rows:
        by_target[int(row["target_mH"])].append(float(row["bounded_loss_u"]))
        by_seed[int(row["seed"])].append(float(row["bounded_loss_u"]))
    trimmed = [np.mean(sorted(by_target[t])[1:-1]) for t in TARGETS]
    seed_means = {s: float(np.mean(v)) for s, v in by_seed.items()}
    first = [
        int(row["first_feasible_evaluation"])
        for row in rows
        if row["first_feasible_evaluation"]
    ]
    return {
        "cases": len(rows),
        "qualification_rate": float(
            np.mean([row["qualified"] == "True" for row in rows])
        ),
        "exact_optimum_cases": int(np.sum(values <= 1e-12)),
        "exact_optimum_rate": float(np.mean(values <= 1e-12)),
        "all_case_mean_u": float(values.mean()),
        "targetwise_trimmed_mean_u": float(np.mean(trimmed)),
        "worst_seed_mean_u": max(seed_means.values()),
        "worst_seed": max(seed_means, key=seed_means.get),
        "median_first_feasible_evaluation": float(np.median(first)),
        "maximum_first_feasible_evaluation": max(first),
    }


def analyze(rows):
    baseline_path = (
        ROOT
        / "results"
        / "coil_2d_relative_tolerance"
        / "cei_budget27"
        / "cei_results.csv"
    )
    baseline = {
        (int(r["seed"]), int(float(r["target_mH"]))): r
        for r in read_rows(baseline_path)
    }
    challenger = {(int(r["seed"]), int(r["target_mH"])): r for r in rows}
    keys = sorted(challenger)
    if any(key not in baseline for key in keys):
        raise RuntimeError("baseline is missing paired cases")
    delta = np.asarray(
        [
            float(challenger[k]["bounded_loss_u"])
            - float(baseline[k]["bounded_loss_u"])
            for k in keys
        ]
    )
    rng = np.random.default_rng(20261012)
    matrix = delta.reshape(len(SEEDS), len(TARGETS)).T
    draws = rng.integers(0, len(TARGETS), (100_000, len(TARGETS)))
    boot = matrix[draws].mean(axis=(1, 2))
    nonzero = delta[delta != 0]
    return {
        "paired_mean_difference_u": float(delta.mean()),
        "target_cluster_bootstrap_95pct_ci": np.quantile(
            boot, [0.025, 0.975]
        ).tolist(),
        "paired_case_wins": int(np.sum(delta < 0)),
        "paired_case_ties": int(np.sum(delta == 0)),
        "paired_case_losses": int(np.sum(delta > 0)),
        "wilcoxon_p": None if len(nonzero) == 0 else float(wilcoxon(delta).pvalue),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "center_index": CENTER_ID,
        "center_point": POINTS[CENTER_ID],
        "seeds": list(SEEDS),
        "targets_mH": list(TARGETS),
        "relative_tolerance": EPSILON,
        "budget": {
            "initial": 1,
            "sequential": 16,
            "confirmation": 10,
            "total": 27,
        },
        "pre_feasible_acquisition": "probability of feasibility",
        "post_feasible_acquisition": "ordinary constrained expected improvement",
        "kernel_policy": "fixed prior for 1--3 observations; baseline optimizer from 4",
        "confirmation": "same GP-only p_feasible*exp(-unit(mu_R)) top-10 rule",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    existing = read_rows(ROWS)
    done = {(int(r["seed"]), int(r["target_mH"])) for r in existing}
    tasks = [(s, t) for s in SEEDS for t in TARGETS if (s, t) not in done]
    new_file = not ROWS.exists()
    with ROWS.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, task): task for task in tasks}
            for count, future in enumerate(as_completed(futures), 1):
                row = future.result()
                writer.writerow(row)
                stream.flush()
                if count % 25 == 0 or count == len(tasks):
                    print(f"checkpoint {count}/{len(tasks)}", flush=True)
    rows = read_rows(ROWS)
    if len(rows) != len(SEEDS) * len(TARGETS):
        raise RuntimeError(f"incomplete results: {len(rows)} rows")
    report = {
        "center_start": metrics(rows),
        "paired_vs_random_six_baseline": analyze(rows),
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest["status"] = "complete"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
