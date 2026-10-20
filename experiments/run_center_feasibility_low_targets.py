import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_center_start_control as base

TARGETS = (2, 3, 4, 5)
WEIGHT = 0.25
OUT = base.ROOT / "results" / "validation" / "center_feasibility_low_targets"
ROWS = OUT / "results.csv"
FIELDS = ("pre_uncertainty_weight",) + base.FIELDS


def read(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    existing = read(ROWS)
    done = {(int(r["seed"]), int(r["target_mH"])) for r in existing}
    tasks = [
        (s, t, WEIGHT)
        for s in base.SEEDS
        for t in TARGETS
        if (s, t) not in done
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
                writer.writerow({"pre_uncertainty_weight": WEIGHT, **row})
                stream.flush()
                print(f"checkpoint {count}/{len(tasks)}", flush=True)
    rows = read(ROWS)
    if len(rows) != len(base.SEEDS) * len(TARGETS):
        raise RuntimeError(f"incomplete targeted test: {len(rows)} rows")
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
    values = np.asarray([float(r["bounded_loss_u"]) for r in rows])
    base_values = np.asarray(
        [
            float(baseline[int(r["seed"]), int(r["target_mH"])]["bounded_loss_u"])
            for r in rows
        ]
    )
    first = [
        int(r["first_feasible_evaluation"])
        for r in rows
        if r["first_feasible_evaluation"]
    ]
    by_target = {}
    for target in TARGETS:
        group = [r for r in rows if int(r["target_mH"]) == target]
        b = [baseline[int(r["seed"]), target] for r in group]
        by_target[str(target)] = {
            "challenger_exact": sum(
                float(r["bounded_loss_u"]) <= 1e-12 for r in group
            ),
            "baseline_exact": sum(float(r["bounded_loss_u"]) <= 1e-12 for r in b),
            "challenger_mean_u": float(
                np.mean([float(r["bounded_loss_u"]) for r in group])
            ),
            "baseline_mean_u": float(
                np.mean([float(r["bounded_loss_u"]) for r in b])
            ),
        }
    summary = {
        "targets_mH": list(TARGETS),
        "seeds": list(base.SEEDS),
        "pre_uncertainty_weight": WEIGHT,
        "budget": {"initial": 1, "sequential": 16, "confirmation": 10, "total": 27},
        "qualification": f"{sum(r['qualified'] == 'True' for r in rows)}/{len(rows)}",
        "exact_optima": f"{int(np.sum(values <= 1e-12))}/{len(rows)}",
        "baseline_exact_optima": f"{int(np.sum(base_values <= 1e-12))}/{len(rows)}",
        "mean_u": float(values.mean()),
        "baseline_mean_u": float(base_values.mean()),
        "paired_wins_ties_losses": [
            int(np.sum(values < base_values)),
            int(np.sum(values == base_values)),
            int(np.sum(values > base_values)),
        ],
        "median_first_feasible_evaluation": float(np.median(first)),
        "maximum_first_feasible_evaluation": max(first),
        "by_target": by_target,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "script": str(Path(__file__).resolve().relative_to(base.ROOT)),
                "targets_mH": list(TARGETS),
                "seeds": list(base.SEEDS),
                "pre_uncertainty_weight": WEIGHT,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
