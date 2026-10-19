import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    data = Path(__file__).resolve().parents[1] / "data"
    folder = data / "wpt_8d_multitarget_validation_uncapped"
    rows = read_rows(folder / "results.csv")
    groups = {
        policy: sorted(
            [row for row in rows if row["policy"] == policy],
            key=lambda row: float(row["target_M_nH"]),
        )
        for policy in {row["policy"] for row in rows}
    }
    auto = groups["automated"]
    conv = groups["conventional_cei_matched"]
    fixed = groups["forced_b50"]
    assert len(auto) == 21
    for row in auto:
        assert row["stopped_naturally"] == "True"
        assert int(row["n_model_building"]) >= 17
        assert int(row["n_confirmation"]) == 10
        assert float(row["stop_rho"]) <= 0.01
        assert int(row["total_evaluations"]) == int(row["n_model_building"]) + 10
    for auto_row, conv_row in zip(auto, conv):
        assert auto_row["total_evaluations"] == conv_row["total_evaluations"]
    auto_r = np.array([float(row["best_R_ohm"]) for row in auto])
    conv_r = np.array([float(row["best_R_ohm"]) for row in conv])
    fixed_r = np.array([float(row["best_R_ohm"]) for row in fixed])
    delta = auto_r - conv_r
    calls = np.array([int(row["total_evaluations"]) for row in auto])
    result = {
        "automatic_mean_calls": float(calls.mean()),
        "automatic_mean_R": float(auto_r.mean()),
        "conventional_mean_R": float(conv_r.mean()),
        "fixed50_mean_R": float(fixed_r.mean()),
        "savings_percent": float(100 * (1 - calls.mean() / 50)),
        "resistance_penalty_percent": float(100 * (auto_r.mean() / fixed_r.mean() - 1)),
        "wins_ties_losses": [
            int((delta < -1e-10).sum()),
            int((np.abs(delta) <= 1e-10).sum()),
            int((delta > 1e-10).sum()),
        ],
        "wilcoxon_p": float(wilcoxon(delta).pvalue),
        "feasible_counts": {
            policy: sum(row["qualified"] == "True" for row in group)
            for policy, group in groups.items()
        },
        "changed_automatic_runs": [],
    }
    (folder / "uncapped_comparison_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
