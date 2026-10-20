import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_center_start_control as base

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "validation" / "center_stopping_heldout_73" / "results.csv"
OUT = ROOT / "results" / "validation" / "stopping_gp_calibration_heldout_73"


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def correlation(error, sigma):
    if len(error) < 2 or np.std(error) == 0 or np.std(sigma) == 0:
        return float("nan")
    return float(np.corrcoef(error, sigma)[0, 1])


def evaluate(row):
    target = int(row["target_mH"])
    selected = np.asarray(json.loads(row["model_building_ids"]), int)
    chosen = set(map(int, selected))
    unseen = np.asarray([i for i in range(len(base.POINTS)) if i not in chosen], int)
    mean_l, std_l = base.fit(
        base.L_ALL, selected, int(row["seed"]), final=True, deterministic_starts=5
    )
    mean_r, std_r = base.fit(
        base.R_ALL, selected, int(row["seed"]), final=True, deterministic_starts=5
    )
    err_l = np.abs(mean_l[unseen] - base.L_ALL[unseen])
    sig_l = np.maximum(std_l[unseen], 1e-12)
    err_r = np.abs(mean_r[unseen] - base.R_ALL[unseen])
    sig_r = np.maximum(std_r[unseen], 1e-12)
    true_feasible = np.abs(base.L_ALL[unseen] - target) / target <= base.EPSILON
    lower, upper = target * (1 - base.EPSILON), target * (1 + base.EPSILON)
    pf = ndtr((upper - mean_l[unseen]) / sig_l) - ndtr((lower - mean_l[unseen]) / sig_l)
    observed_feasible = selected[
        np.abs(base.L_ALL[selected] - target) / target <= base.EPSILON
    ]
    if not len(observed_feasible):
        raise RuntimeError("held-out stopping history has no feasible incumbent")
    incumbent = float(np.min(base.R_ALL[observed_feasible]))
    domain = np.flatnonzero(np.abs(base.L_ALL - target) / target <= base.EPSILON)
    optimum = float(np.min(base.R_ALL[domain]))
    available = max(0.0, (incumbent - optimum) / incumbent)
    return {
        "target_mH": target,
        "n_model_building": len(selected),
        "final_rho": float(row["final_rho"]),
        "incumbent_R_at_stop": incumbent,
        "exhaustive_R": optimum,
        "actual_available_relative_improvement": available,
        "incumbent_exact_at_stop": abs(incumbent - optimum) <= 1e-12,
        "all_n": len(unseen),
        "all_L_mae": float(np.mean(err_l)),
        "all_L_coverage_1sigma": float(np.mean(err_l <= sig_l)),
        "all_L_coverage_2sigma": float(np.mean(err_l <= 2 * sig_l)),
        "all_L_error_sigma_correlation": correlation(err_l, sig_l),
        "all_R_mae": float(np.mean(err_r)),
        "all_R_coverage_1sigma": float(np.mean(err_r <= sig_r)),
        "all_R_coverage_2sigma": float(np.mean(err_r <= 2 * sig_r)),
        "all_R_error_sigma_correlation": correlation(err_r, sig_r),
        "feasible_n": int(np.sum(true_feasible)),
        "feasible_L_mae": float(np.mean(err_l[true_feasible])),
        "feasible_L_coverage_1sigma": float(
            np.mean(err_l[true_feasible] <= sig_l[true_feasible])
        ),
        "feasible_L_coverage_2sigma": float(
            np.mean(err_l[true_feasible] <= 2 * sig_l[true_feasible])
        ),
        "feasible_R_mae": float(np.mean(err_r[true_feasible])),
        "feasible_R_coverage_1sigma": float(
            np.mean(err_r[true_feasible] <= sig_r[true_feasible])
        ),
        "feasible_R_coverage_2sigma": float(
            np.mean(err_r[true_feasible] <= 2 * sig_r[true_feasible])
        ),
        "feasibility_brier": float(np.mean((pf - true_feasible.astype(float)) ** 2)),
    }


def main():
    source = read_rows(INPUT)
    OUT.mkdir(parents=True, exist_ok=True)
    output = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(evaluate, row): row for row in source}
        for count, future in enumerate(as_completed(futures), 1):
            output.append(future.result())
            if count % 10 == 0 or count == len(source):
                print(f"checkpoint {count}/{len(source)}", flush=True)
    output.sort(key=lambda row: row["target_mH"])
    with (OUT / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    def macro(name):
        values = np.asarray([row[name] for row in output], float)
        return float(np.nanmean(values))

    summary = {
        "cases": len(output),
        "evaluation_point": "final target-conditioned GP at held-out stopping point, before confirmations",
        "all_unseen_grid_macro_mean": {
            key: macro(key)
            for key in (
                "all_L_mae",
                "all_L_coverage_1sigma",
                "all_L_coverage_2sigma",
                "all_L_error_sigma_correlation",
                "all_R_mae",
                "all_R_coverage_1sigma",
                "all_R_coverage_2sigma",
                "all_R_error_sigma_correlation",
                "feasibility_brier",
            )
        },
        "true_feasible_region_macro_mean": {
            key: macro(key)
            for key in (
                "feasible_L_mae",
                "feasible_L_coverage_1sigma",
                "feasible_L_coverage_2sigma",
                "feasible_R_mae",
                "feasible_R_coverage_1sigma",
                "feasible_R_coverage_2sigma",
            )
        },
        "stopping_quality": {
            "incumbent_exact_at_stop": int(
                sum(row["incumbent_exact_at_stop"] for row in output)
            ),
            "mean_actual_available_relative_improvement": macro(
                "actual_available_relative_improvement"
            ),
            "maximum_actual_available_relative_improvement": float(
                max(row["actual_available_relative_improvement"] for row in output)
            ),
            "targets_with_actual_improvement_above_1pct": int(
                sum(row["actual_available_relative_improvement"] > 0.01 for row in output)
            ),
            "mean_final_predicted_rho": macro("final_rho"),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "script": str(Path(__file__).resolve().relative_to(ROOT)),
                "input": str(INPUT.relative_to(ROOT)),
                "targets": len(output),
                "configuration_frozen": True,
                "confirmations_excluded": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
