import csv
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "aedt_wpt_1000_matched"
AUTO_ZIP = DATA / "automatic_exact_run.zip"
BASE_ZIP = DATA / "sobol6_cei14_baseline.zip"
DESIGN = (
    "ns_t",
    "nt_t",
    "s_t_mm",
    "ri_t_mm",
    "ns_r",
    "nt_r",
    "s_r_mm",
    "ri_r_mm",
)


def close(actual, expected, tolerance=1e-10):
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual!r} != {expected!r}")


def read_csv(archive, name):
    text = archive.read(name).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def feasible(row):
    return row["feasible"].strip().lower() == "true"


def design_key(row):
    return tuple(float(row[column]) for column in DESIGN)


def best_feasible(rows):
    return min(
        (row for row in rows if feasible(row)),
        key=lambda row: float(row["J_Rac_sum_ohm"]),
    )


def main():
    with zipfile.ZipFile(AUTO_ZIP) as outer:
        auto_rows = read_csv(outer, "trajectory.csv")
        auto_summary = json.loads(outer.read("final_summary.json"))
        nested_bytes = outer.read("exact_auto_wpt_icece_run.zip")
        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
            required = {
                "exact_auto_wpt_icece.py",
                "trajectory.json",
                "confirmation_plan.json",
                "optimizer.log",
            }
            if not required.issubset(nested.namelist()):
                raise AssertionError("automatic archive lacks replay evidence")

    with zipfile.ZipFile(BASE_ZIP) as archive:
        base_rows = read_csv(archive, "trajectory.csv")
        base_summary = json.loads(archive.read("final_summary.json"))
        comparison = json.loads(archive.read("comparison_vs_automatic.json"))
        required = {
            "baseline_sobol6_cei14.py",
            "exact_auto_wpt_icece.py",
            "trajectory.json",
            "confirmation_plan.json",
            "optimizer.log",
        }
        if not required.issubset(archive.namelist()):
            raise AssertionError("baseline archive lacks replay evidence")

    assert len(auto_rows) == len(base_rows) == 30
    assert len({design_key(row) for row in auto_rows}) == 30
    assert len({design_key(row) for row in base_rows}) == 30
    assert Counter(row["phase"] for row in auto_rows) == {
        "center": 1,
        "feasibility": 11,
        "cei": 8,
        "confirmation": 10,
    }
    assert Counter(row["phase"] for row in base_rows) == {
        "initial_sobol": 6,
        "sequential_cei": 14,
        "confirmation": 10,
    }
    assert sum(feasible(row) for row in auto_rows) == 7
    assert sum(feasible(row) for row in base_rows) == 3
    assert next(i for i, row in enumerate(auto_rows, 1) if feasible(row)) == 12
    assert next(i for i, row in enumerate(base_rows, 1) if feasible(row)) == 6

    auto_best = best_feasible(auto_rows)
    base_best = best_feasible(base_rows)
    close(float(auto_best["mutual_nH"]), 998.929041993277)
    close(float(auto_best["J_Rac_sum_ohm"]), 0.3836807971)
    close(float(base_best["mutual_nH"]), 1016.4019754008999)
    close(float(base_best["J_Rac_sum_ohm"]), 0.3895351708)
    close(float(auto_best["mesh_max_length_mm"]), 1.0)
    close(float(base_best["mesh_max_length_mm"]), 0.0)

    assert auto_summary["n_model_building"] == 20
    assert auto_summary["n_confirmation"] == 10
    assert auto_summary["stopped_naturally"] is True
    close(float(auto_summary["final_rho"]), 0.00877666862945453)
    assert base_summary["n_initial"] == 6
    assert base_summary["n_sequential"] == 14
    assert base_summary["n_confirmation"] == 10
    assert comparison["same_total_budget"] is True

    delta = float(base_best["J_Rac_sum_ohm"]) - float(auto_best["J_Rac_sum_ohm"])
    reduction = 100.0 * delta / float(base_best["J_Rac_sum_ohm"])
    close(delta, 0.0058543737)
    close(reduction, 1.5029127377578433, tolerance=1e-9)

    print(
        json.dumps(
            {
                "status": "PASS",
                "accepted_evaluations_each": 30,
                "automatic": {
                    "first_feasible_call": 12,
                    "feasible_observations": 7,
                    "best_M_nH": float(auto_best["mutual_nH"]),
                    "best_J_ohm": float(auto_best["J_Rac_sum_ohm"]),
                },
                "sobol6_cei14": {
                    "first_feasible_call": 6,
                    "feasible_observations": 3,
                    "best_M_nH": float(base_best["mutual_nH"]),
                    "best_J_ohm": float(base_best["J_Rac_sum_ohm"]),
                },
                "automatic_J_reduction_percent_vs_baseline": reduction,
                "mesh_caveat_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
