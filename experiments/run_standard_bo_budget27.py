import argparse
import contextlib
import csv
import json
import os
import runpy
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_surrogate_targets import ROOT, truth_grid, write_json

SEEDS = tuple(range(20261015, 20261025))
TARGETS = tuple(range(1, 88))
EPSILON = 0.025
BUDGET = 27
MODEL_EVALUATIONS = 17
FINAL_VERIFICATIONS = 10
FIELDS = (
    "method",
    "budget",
    "seed",
    "target_mH",
    "qualified",
    "relative_resistance_regret",
    "bounded_loss_u",
    "decision_T",
    "decision_n",
    "decision_L_mH",
    "decision_R_ohm",
    "decision_source",
)


def feasible(point, target):
    return abs(point["L_mH"] - target) / target <= EPSILON


def run_one(task):
    mode, seed, target, local_weight, global_weight, alpha = task
    with tempfile.TemporaryDirectory(prefix=f"coil-{mode}-") as directory:
        workspace = Path(directory)
        shutil.copytree(ROOT / "config", workspace / "config")
        shutil.copytree(ROOT / "methods", workspace / "methods")
        (workspace / "results").mkdir()
        settings_path = workspace / "config" / "surrogates" / "target_conditioned_gp.json"
        settings = json.loads(settings_path.read_text())
        settings.update(
            {
                "seed": seed,
                "target_mH": target,
                "acquisition_mode": mode,
                "local_weight": local_weight,
                "global_weight": global_weight,
                "global_response_alpha": alpha,
                "maximum_true_evaluations": MODEL_EVALUATIONS,
                "final_validation_count": FINAL_VERIFICATIONS,
            }
        )
        write_json(settings_path, settings)
        script = (
            workspace
            / "methods"
            / "target_informed_gp"
            / "gp_conventional_acquisition_baselines.py"
        )
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            values = runpy.run_path(str(script), run_name="__main__")
        decision = values["decision"]
        config = json.loads((workspace / "config" / "coil_problem.json").read_text())
        grid = truth_grid(config)
        optimum = min(
            (point for point in grid if feasible(point, target)),
            key=lambda point: point["R_ohm"],
        )
        qualified = feasible(decision, target)
        regret = decision["R_ohm"] / optimum["R_ohm"] - 1 if qualified else None
        loss = regret / (1 + regret) if qualified else 1.0
        return {
            "method": mode,
            "budget": BUDGET,
            "seed": seed,
            "target_mH": target,
            "qualified": qualified,
            "relative_resistance_regret": regret,
            "bounded_loss_u": loss,
            "decision_T": decision["point"][0],
            "decision_n": decision["point"][1],
            "decision_L_mH": decision["L_mH"],
            "decision_R_ohm": decision["R_ohm"],
            "decision_source": decision["source"],
        }


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


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
        "exact_optimum_cases": sum(value <= 1e-12 for value in losses),
        "exact_optimum_rate": sum(value <= 1e-12 for value in losses) / len(losses),
        "all_case_mean_u": sum(losses) / len(losses),
        "targetwise_trimmed_mean_u": sum(trimmed) / len(trimmed),
        "worst_seed_mean_u": max(seed_means.values()),
        "worst_seed": max(seed_means, key=seed_means.get),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("cei", "clcb", "ivr"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--targets", nargs="+", type=int, default=TARGETS)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--global-weight", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    if (
        args.local_weight < 0
        or args.global_weight < 0
        or abs(args.local_weight + args.global_weight - 1) > 1e-12
    ):
        parser.error("local and global weights must be nonnegative and sum to one")
    if not 0 <= args.alpha <= 1:
        parser.error("alpha must lie in [0, 1]")
    targets = tuple(args.targets)
    tag = (
        f"{args.mode}_wl{args.local_weight:g}_wg{args.global_weight:g}_a{args.alpha:g}".replace(
            ".", "p"
        )
    )
    output = (
        ROOT
        / "results"
        / "coil_2d_relative_tolerance"
        / "standard_bo_budget27"
        / tag
    )
    results = output / "results.csv"
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "manifest.json",
        {
            "status": "running",
            "mode": args.mode,
            "budget": BUDGET,
            "seeds": list(SEEDS),
            "targets_mH": list(targets),
            "local_weight": args.local_weight,
            "global_weight": args.global_weight,
            "global_response_alpha": args.alpha,
            "budget_composition": "6 initial + 11 acquisitions + 10 final verifications",
        },
    )
    completed = {
        (int(row["seed"]), int(float(row["target_mH"]))) for row in read_rows(results)
    }
    tasks = [
        (args.mode, seed, target, args.local_weight, args.global_weight, args.alpha)
        for seed in SEEDS
        for target in targets
        if (seed, target) not in completed
    ]
    new_file = not results.exists()
    with results.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                writer.writerow(row)
                stream.flush()
                print(
                    f"checkpoint {index}/{len(tasks)} seed={row['seed']} target={row['target_mH']}",
                    flush=True,
                )
    selected = [
        row for row in read_rows(results) if int(float(row["target_mH"])) in targets
    ]
    if len(selected) != len(SEEDS) * len(targets):
        raise RuntimeError("incomplete benchmark")
    summary = metrics(selected, targets)
    write_json(output / "summary.json", summary)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
