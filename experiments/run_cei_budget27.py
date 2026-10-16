"""Matched B=27 conventional GP-cEI experiment over 10 seeds and 87 targets.

The kernel, initial designs, target tolerance, final validation rule, and final
decision pool are identical to TC-FEI.  Only the acquisition is changed to
conventional constrained expected improvement (p_feasible * EI_R), with
probability of feasibility used until a feasible incumbent is observed.
"""
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

from compare_surrogate_targets import ROOT, truth_grid, write_json

SEEDS = tuple(range(20261015, 20261025))
EPSILON = 0.025
BUDGET = 27
MODEL_EVALUATIONS = 17
FINAL_VERIFICATIONS = 10
OUTPUT = (ROOT / "results" / "coil_2d_relative_tolerance" /
          "cei_budget27")
RESULTS = OUTPUT / "cei_results.csv"
FIELDS = ("method", "budget", "seed", "target_mH", "qualified",
          "relative_resistance_regret", "bounded_loss_u", "decision_T",
          "decision_n", "decision_L_mH", "decision_R_ohm", "decision_source")


def feasible(point, target):
    return abs(point["L_mH"] - target) / target <= EPSILON


def run_one(task):
    seed, target = task
    with tempfile.TemporaryDirectory(prefix="coil-cei-") as directory:
        workspace = Path(directory)
        shutil.copytree(ROOT / "config", workspace / "config")
        shutil.copytree(ROOT / "methods", workspace / "methods")
        (workspace / "results").mkdir()
        settings_path = workspace / "config" / "surrogates" / "target_conditioned_gp.json"
        settings = json.loads(settings_path.read_text())
        settings.update({
            "seed": seed,
            "target_mH": target,
            "acquisition_mode": "cei",
            "maximum_true_evaluations": MODEL_EVALUATIONS,
            "final_validation_count": FINAL_VERIFICATIONS,
        })
        write_json(settings_path, settings)
        script = workspace / "methods" / "target_informed_gp" / "gp_conventional_acquisition_baselines.py"
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            values = runpy.run_path(str(script), run_name="__main__")
        decision = values["decision"]
        config = json.loads((workspace / "config" / "coil_problem.json").read_text())
        grid = truth_grid(config)
        optimum = min((point for point in grid if feasible(point, target)), key=lambda point: point["R_ohm"])
        qualified = feasible(decision, target)
        regret = decision["R_ohm"] / optimum["R_ohm"] - 1 if qualified else None
        loss = regret / (1 + regret) if qualified else 1.0
        return {
            "method": "cei", "budget": BUDGET, "seed": seed, "target_mH": target,
            "qualified": qualified, "relative_resistance_regret": regret,
            "bounded_loss_u": loss, "decision_T": decision["point"][0],
            "decision_n": decision["point"][1], "decision_L_mH": decision["L_mH"],
            "decision_R_ohm": decision["R_ohm"], "decision_source": decision["source"],
        }


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def append_row(row):
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def metrics(rows, targets):
    losses = [float(row["bounded_loss_u"]) for row in rows]
    qualified = [row["qualified"].lower() == "true" for row in rows]
    by_seed, by_target = defaultdict(list), defaultdict(list)
    for row in rows:
        value = float(row["bounded_loss_u"])
        by_seed[int(row["seed"])].append(value)
        by_target[int(float(row["target_mH"]))].append(value)
    trimmed = [sum(sorted(by_target[target])[1:-1]) / 8 for target in targets]
    seed_means = {seed: sum(values) / len(values) for seed, values in by_seed.items()}
    return {
        "cases": len(rows),
        "qualification_rate": sum(qualified) / len(qualified),
        "exact_optimum_rate": sum(value <= 1e-12 for value in losses) / len(losses),
        "exact_optimum_cases": sum(value <= 1e-12 for value in losses),
        "all_case_mean_u": sum(losses) / len(losses),
        "targetwise_trimmed_mean_u": sum(trimmed) / len(trimmed),
        "worst_seed_mean_u": max(seed_means.values()),
        "worst_seed": max(seed_means, key=seed_means.get),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--targets", nargs="+", type=int, default=range(1, 88))
    args = parser.parse_args()
    targets = tuple(args.targets)
    if any(target < 1 or target > 87 for target in targets):
        parser.error("targets must be integer values from 1 to 87 mH")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "manifest.json", {
        "status": "running", "method": "conventional GP-cEI", "budget": BUDGET,
        "seeds": list(SEEDS), "targets_mH": list(targets),
        "relative_tolerance": EPSILON,
        "budget_composition": "6 initial + 11 cEI acquisitions + 10 final verifications",
        "matched_control_for": "TC-FEI",
    })
    completed = {(int(row["seed"]), int(float(row["target_mH"]))) for row in read_rows(RESULTS)}
    tasks = [(seed, target) for seed in SEEDS for target in targets if (seed, target) not in completed]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_row(row)
            print(f"checkpoint {index}/{len(tasks)}: seed={row['seed']} target={row['target_mH']}", flush=True)
    rows = read_rows(RESULTS)
    selected = [row for row in rows if int(float(row["target_mH"])) in targets]
    if len(selected) != len(SEEDS) * len(targets):
        raise RuntimeError("incomplete cEI data")
    summary = metrics(selected, targets)
    source_counts = defaultdict(int)
    for row in selected:
        source_counts[row["decision_source"]] += 1
    summary["decision_sources"] = dict(source_counts)
    write_json(OUTPUT / "summary.json", summary)
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    manifest["status"] = "complete"
    write_json(OUTPUT / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
