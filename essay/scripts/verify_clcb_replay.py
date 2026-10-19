from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import run_standard_bo_budget27 as benchmark


OUT = ROOT / "essay" / "data" / "clcb_correct_sign_replay"
SOURCE = ROOT / "methods" / "target_informed_gp" / "gp_conventional_acquisition_baselines.py"
REFERENCE = (ROOT / "results" / "coil_2d_relative_tolerance" /
             "standard_bo_budget27" / "clcb_wl1_wg0_a0p5" / "results.csv")


def load(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [("clcb", seed, target, 1., 0., .5)
             for seed in benchmark.SEEDS for target in benchmark.TARGETS]
    result_path = OUT / "results.csv"
    replay = load(result_path) if result_path.exists() else []
    if len(replay) != len(tasks):
        replay = []
        with ProcessPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(benchmark.run_one, task): task for task in tasks}
            for count, future in enumerate(as_completed(futures), 1):
                replay.append(future.result())
                if count % 50 == 0 or count == len(tasks):
                    print("checkpoint %d/%d" % (count, len(tasks)), flush=True)
        replay.sort(key=lambda row: (int(row["seed"]), int(row["target_mH"])))
        with result_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=benchmark.FIELDS)
            writer.writeheader(); writer.writerows(replay)
        replay = load(result_path)
    summary = benchmark.metrics(replay, benchmark.TARGETS)
    reference = {(int(row["seed"]), int(float(row["target_mH"]))): row
                 for row in load(REFERENCE)}
    mismatches = []
    for row in replay:
        key = int(row["seed"]), int(row["target_mH"])
        old = reference[key]
        for field in ("qualified", "decision_T", "decision_n", "decision_source"):
            if str(row[field]) != old[field]:
                mismatches.append([key, field, old[field], row[field]])
        for field in ("decision_L_mH", "decision_R_ohm", "bounded_loss_u"):
            if abs(float(row[field])-float(old[field])) > 2e-13:
                mismatches.append([key, field, old[field], row[field]])
    code = SOURCE.read_bytes()
    report = {
        "status": "complete" if not mismatches else "mismatch",
        "implementation": {
            "lower_bound": "mean_R - kappa*std_R",
            "kappa": 2.0,
            "maximized_acquisition": "p_feasible*exp(-unit_scale(lower_bound))",
            "source": str(SOURCE),
            "source_sha256": hashlib.sha256(code).hexdigest(),
        },
        "fresh_replay_summary": summary,
        "reference_rows": len(reference),
        "fresh_rows": len(replay),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit("fresh cLCB replay differs from archived Fig. 4 data")


if __name__ == "__main__":
    main()
