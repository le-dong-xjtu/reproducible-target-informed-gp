from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import runpy
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from compare_surrogate_targets import ROOT, truth_grid, write_json

SEEDS = tuple(range(20261015, 20261025))
TARGETS = tuple(range(1, 88))
EPSILON = 0.025
BUDGET = 27
MODEL_EVALUATIONS = 17
VALIDATIONS = 10
FIELDS = ("method", "budget", "seed", "target_mH", "qualified",
          "relative_resistance_regret", "bounded_loss_u", "decision_T",
          "decision_n", "decision_L_mH", "decision_R_ohm")


def feasible(point, target):
    return abs(point["L_mH"] - target) / target <= EPSILON


def unit_scale(values):
    values = np.asarray(values, dtype=float)
    span = np.ptp(values)
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def run_seed(task):
    seed, alpha = task
    with tempfile.TemporaryDirectory(prefix="coil-global-dual-") as directory:
        workspace = Path(directory)
        shutil.copytree(ROOT / "config", workspace / "config")
        shutil.copytree(ROOT / "methods", workspace / "methods")
        (workspace / "results").mkdir()
        settings_path = workspace / "config" / "surrogates" / "global_dual_gp.json"
        settings = json.loads(settings_path.read_text())
        settings.update({
            "seed": seed,
            "alpha": alpha,
            "maximum_true_evaluations": MODEL_EVALUATIONS,
            "final_validation_count": VALIDATIONS,
        })
        write_json(settings_path, settings)
        script = workspace / "methods" / "surrogate_selection" / "gp_dual_response.py"
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            values = runpy.run_path(str(script), run_name="__main__")
        config = json.loads((workspace / "config" / "coil_problem.json").read_text())
        grid = truth_grid(config)
        seen = set(values["seen"])
        mean = np.asarray(values["mean"])
        std = np.maximum(np.asarray(values["std"]), 1e-12)
        mean_R = np.asarray(values["mean_R"])
        unseen = np.array([index for index in range(len(grid)) if index not in seen])
        rows = []
        for target in TARGETS:
            lower, upper = target * (1 - EPSILON), target * (1 + EPSILON)
            p_feasible = (ndtr((upper - mean[unseen]) / std[unseen])
                          - ndtr((lower - mean[unseen]) / std[unseen]))
            validation_score = p_feasible * np.exp(-unit_scale(mean_R[unseen]))
            final_ids = unseen[np.argsort(-validation_score)[:VALIDATIONS]]
            candidates = [grid[int(index)] for index in final_ids]
            valid = [point for point in candidates if feasible(point, target)]
            decision = min(valid, key=lambda point: point["R_ohm"]) if valid else min(
                candidates, key=lambda point: abs(point["L_mH"] - target) / target
            )
            optimum = min((point for point in grid if feasible(point, target)), key=lambda point: point["R_ohm"])
            qualified = feasible(decision, target)
            regret = decision["R_ohm"] / optimum["R_ohm"] - 1 if qualified else None
            loss = regret / (1 + regret) if qualified else 1.0
            rows.append({
                "method": "global_dual_gp", "budget": BUDGET, "seed": seed,
                "target_mH": target, "qualified": qualified,
                "relative_resistance_regret": regret, "bounded_loss_u": loss,
                "decision_T": decision["T"], "decision_n": decision["n"],
                "decision_L_mH": decision["L_mH"], "decision_R_ohm": decision["R_ohm"],
            })
        return rows


def metrics(rows):
    losses = [float(row["bounded_loss_u"]) for row in rows]
    by_seed, by_target = defaultdict(list), defaultdict(list)
    for row in rows:
        value = float(row["bounded_loss_u"])
        by_seed[int(row["seed"])].append(value)
        by_target[int(float(row["target_mH"]))].append(value)
    seed_means = {seed: sum(values) / len(values) for seed, values in by_seed.items()}
    trimmed = [sum(sorted(by_target[target])[1:-1]) / 8 for target in TARGETS]
    return {
        "cases": len(rows),
        "qualification_rate": sum(row["qualified"].lower() == "true" for row in rows) / len(rows),
        "exact_optimum_cases": sum(value <= 1e-12 for value in losses),
        "exact_optimum_rate": sum(value <= 1e-12 for value in losses) / len(losses),
        "all_case_mean_u": sum(losses) / len(losses),
        "targetwise_trimmed_mean_u": sum(trimmed) / len(trimmed),
        "worst_seed_mean_u": max(seed_means.values()),
        "worst_seed": max(seed_means, key=seed_means.get),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("alpha must lie in [0, 1]")
    tag = f"alpha_{args.alpha:g}".replace(".", "p")
    output = (ROOT / "results" / "coil_2d_relative_tolerance" /
              "global_dual_gp_budget27" / tag)
    results = output / "results.csv"
    output.mkdir(parents=True, exist_ok=True)
    completed = set()
    if results.exists():
        with results.open(newline="") as stream:
            completed = {int(row["seed"]) for row in csv.DictReader(stream)}
    tasks = [(seed, args.alpha) for seed in SEEDS if seed not in completed]
    new_file = not results.exists()
    with results.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_seed, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                rows = future.result()
                writer.writerows(rows)
                stream.flush()
                print(f"checkpoint {index}/{len(tasks)} seed={rows[0]['seed']}", flush=True)
    with results.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = metrics(rows)
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {
        "status": "complete", "alpha": args.alpha, "seeds": list(SEEDS),
        "targets_mH": list(TARGETS),
        "acquisition": "(1-alpha)*sigma_L/s_L + alpha*sigma_R/s_R",
        "scales": "fixed after the six initial observations",
        "budget_composition": "17 reusable model-building evaluations + 10 final verifications per target",
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
