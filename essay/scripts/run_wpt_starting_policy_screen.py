from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run_automatic_gp_wpt8d as auto
from run_wpt_8d_multitarget_validation import components, confirmation

OUT = auto.PROJECT / "essay/data/wpt_starting_policy_screen_uncapped"
TARGET = 1500.0
SEEDS = tuple(range(20261015, 20261025))


def run(seed=None):
    name = "deterministic" if seed is None else f"random6_{seed}"
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    stream = auto.sobol_stream()
    if seed is None:
        x = np.asarray([auto.CENTER])
    else:
        rng = np.random.default_rng(seed)
        points = []
        while len(points) < 6:
            point = rng.uniform(auto.LOW, auto.HIGH)
            for index in (0, 1, 4, 5):
                point[index] = rng.integers(
                    int(auto.LOW[index]), int(auto.HIGH[index]) + 1
                )
            if not any(np.array_equal(point, previous) for previous in points):
                points.append(point)
        x = np.asarray(points)
    y = np.asarray([auto.evaluate(point) for point in x])
    initial = len(x)
    streak, stopped = 0, False
    refits = []
    while True:
        step = len(x) - 1
        candidates = auto.unseen_unique(
            stream[1 + step * auto.N_CANDIDATES : 1 + (step + 1) * auto.N_CANDIDATES], x
        )
        models = auto.gps(auto.normalize(x), y)
        pf, _, sd, cei, incumbent = components(
            models, auto.normalize(candidates), y, TARGET
        )
        rho = None
        if incumbent is None:
            score = auto.FEASIBILITY_WEIGHT * auto.unit(pf) + (
                1 - auto.FEASIBILITY_WEIGHT
            ) * auto.unit(sd)
        else:
            rho = float(np.max(cei))
            streak = streak + 1 if rho <= auto.STOP_DELTA else 0
            score = cei
        refits.append({"observations": len(x), "rho": rho, "streak": streak})
        if len(x) >= auto.MIN_MODEL_CALLS and streak >= auto.STOP_PATIENCE:
            stopped = True
            break
        chosen = candidates[int(np.argmax(score))]
        x = np.vstack([x, chosen])
        y = np.vstack([y, auto.evaluate(chosen)])
    n_model = len(x)
    xc, yc = confirmation(x, y, TARGET, stream, {"oracle": 0.0, "surrogate": 0.0})
    x, y = np.vstack([x, xc]), np.vstack([y, yc])
    mask = np.abs(y[:, 0] - TARGET) <= auto.EPSILON * TARGET
    valid = np.flatnonzero(mask)
    best = int(valid[np.argmin(y[valid, 1])]) if len(valid) else None
    result = {
        "policy": name,
        "seed": seed,
        "target_M_nH": TARGET,
        "qualified": bool(len(valid)),
        "best_R_ohm": float(y[best, 1]) if best is not None else None,
        "best_M_nH": float(y[best, 0]) if best is not None else None,
        "best_evaluation": best + 1 if best is not None else None,
        "first_feasible_evaluation": int(valid[0]) + 1 if len(valid) else None,
        "n_initial": initial,
        "n_model_building": n_model,
        "n_confirmation": len(xc),
        "total_evaluations": len(x),
        "stopped_naturally": stopped,
    }
    trace = []
    incumbent = None
    for i, (point, response) in enumerate(zip(x, y)):
        if mask[i]:
            incumbent = (
                min(incumbent, float(response[1]))
                if incumbent is not None
                else float(response[1])
            )
        trace.append(
            {
                "evaluation": i + 1,
                "stage": (
                    "initial"
                    if i < initial
                    else "sequential" if i < n_model else "verification"
                ),
                "M_nH": float(response[0]),
                "R_ohm": float(response[1]),
                "feasible": bool(mask[i]),
                "best_feasible_R_ohm": incumbent,
                "design": json.dumps(point.tolist()),
            }
        )
    with (folder / "trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (folder / "refits.json").write_text(json.dumps(refits, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "target_nH": TARGET,
                "random_seeds": SEEDS,
                "selection_rule": "Figure uses user-selected seed 20261019 from the fixed ten-seed screen. Worst final resistance is also recorded for auditing; it is not the plotted seed.",
                "shared_settings": {
                    "omega": auto.FEASIBILITY_WEIGHT,
                    "K": auto.GP_STARTS,
                    "delta": auto.STOP_DELTA,
                    "P": auto.STOP_PATIENCE,
                    "minimum_model_observations": auto.MIN_MODEL_CALLS,
                    "maximum_model_observations": None,
                    "verification_evaluations": auto.N_CONFIRM,
                },
                "random_initialization": "Six uniform mixed-domain points; integer variables sampled uniformly as integers; continuous variables uniformly within bounds; NumPy default_rng.",
                "candidates": "Identical unscrambled Sobol stream and observation-count-indexed blocks; common final pool. Observed candidates excluded separately per run.",
                "scope": "Initialization policy differs in placement and count. Both policies use blended feasibility discovery, deterministic GP fitting, and adaptive stopping. Numerical WPT, not Maxwell.",
            },
            indent=2,
        )
        + "\n"
    )
    with threadpool_limits(limits=1):
        results = [run()]
        reference = json.loads((auto.OUT / "result.json").read_text())
        assert results[0]["total_evaluations"] == reference["total_evaluations"]
        assert np.isclose(results[0]["best_R_ohm"], reference["best_R_ohm"], rtol=1e-8)
        results.extend(run(seed) for seed in SEEDS)
    worst = max(results[1:], key=lambda r: (not r["qualified"], r["best_R_ohm"] or 0))
    (OUT / "summary.json").write_text(
        json.dumps(
            {"results": results, "selected_worst_policy": worst["policy"]}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
