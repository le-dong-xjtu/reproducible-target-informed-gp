import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_center_start_control as base

TARGETS = (1, 11, 20, 30, 39, 49, 58, 68, 77, 87)
WEIGHTS = (0.25, 0.50, 0.75)
OUT = base.ROOT / "results" / "validation" / "center_feasibility_smoke"
ROWS = OUT / "results.csv"
FIELDS = ("pre_uncertainty_weight",) + base.FIELDS


def read(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(rows):
    baseline_rows = read(
        base.ROOT
        / "results"
        / "coil_2d_relative_tolerance"
        / "cei_budget27"
        / "cei_results.csv"
    )
    baseline = {
        (int(r["seed"]), int(float(r["target_mH"]))): r for r in baseline_rows
    }
    result = []
    for label, group in [
        (str(w), [r for r in rows if float(r["pre_uncertainty_weight"]) == w])
        for w in WEIGHTS
    ]:
        keys = [(int(r["seed"]), int(r["target_mH"])) for r in group]
        values = np.asarray([float(r["bounded_loss_u"]) for r in group])
        base_values = np.asarray([float(baseline[k]["bounded_loss_u"]) for k in keys])
        first_feasible = [
            int(r["first_feasible_evaluation"])
            for r in group
            if r["first_feasible_evaluation"]
        ]
        qualified_targets = [
            t
            for t in TARGETS
            if all(
                next(
                    r
                    for r in group
                    if int(r["seed"]) == s and int(r["target_mH"]) == t
                )["qualified"]
                == "True"
                for s in base.SEEDS
            )
        ]
        result.append(
            {
                "pre_uncertainty_weight": label,
                "cases": len(group),
                "qualification_rate": float(
                    np.mean([r["qualified"] == "True" for r in group])
                ),
                "exact_optimum_rate": float(np.mean(values <= 1e-12)),
                "exact_optimum_cases": int(np.sum(values <= 1e-12)),
                "mean_bounded_loss_u": float(values.mean()),
                "mean_difference_vs_random_six_baseline_u": float(
                    (values - base_values).mean()
                ),
                "paired_wins_ties_losses": [
                    int(np.sum(values < base_values)),
                    int(np.sum(values == base_values)),
                    int(np.sum(values > base_values)),
                ],
                "feasible_before_confirmation_cases": len(first_feasible),
                "median_first_feasible_evaluation": (
                    float(np.median(first_feasible)) if first_feasible else None
                ),
                "mean_first_feasible_evaluation": (
                    float(np.mean(first_feasible)) if first_feasible else None
                ),
                "maximum_first_feasible_evaluation": (
                    max(first_feasible) if first_feasible else None
                ),
                "targets_qualifying_all_ten_seeds": qualified_targets,
            }
        )
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    existing = read(ROWS)
    done = {
        (float(r["pre_uncertainty_weight"]), int(r["seed"]), int(r["target_mH"]))
        for r in existing
    }
    tasks = [
        (s, t, w)
        for w in WEIGHTS
        for s in base.SEEDS
        for t in TARGETS
        if (w, s, t) not in done
    ]
    new_file = not ROWS.exists()
    with ROWS.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(base.run_one, task): task for task in tasks}
            for count, future in enumerate(as_completed(futures), 1):
                row = future.result()
                row = {"pre_uncertainty_weight": futures[future][2], **row}
                writer.writerow(row)
                stream.flush()
                if count % 25 == 0 or count == len(tasks):
                    print(f"checkpoint {count}/{len(tasks)}", flush=True)
    rows = read(ROWS)
    expected = len(WEIGHTS) * len(base.SEEDS) * len(TARGETS)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete smoke test: {len(rows)}/{expected}")
    summary = {
        "targets_mH": list(TARGETS),
        "seeds": list(base.SEEDS),
        "budget": {"initial": 1, "sequential": 16, "confirmation": 10, "total": 27},
        "pre_feasibility_score": "(1-w)*unit(p_feasible)+w*unit(sigma_L)",
        "post_feasibility_score": "ordinary cEI",
        "results": summarize(rows),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "script": str(Path(__file__).resolve().relative_to(base.ROOT)),
                "targets_mH": list(TARGETS),
                "weights": list(WEIGHTS),
                "seeds": list(base.SEEDS),
                "uniform_budget": "1 initial + 16 sequential + 10 final verifications",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
