import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def main():
    data = Path(__file__).resolve().parents[1] / "data"
    folder = data / "wpt_8d_multitarget_validation_uncapped"
    rows = list(csv.DictReader((folder / "results.csv").open()))
    old_path = data / "wpt_8d_multitarget_validation" / "results.csv"
    old = list(csv.DictReader(old_path.open())) if old_path.exists() else []
    groups = {p: sorted([r for r in rows if r["policy"] == p],
                        key=lambda r: float(r["target_M_nH"]))
              for p in {r["policy"] for r in rows}}
    auto = groups["automated"]
    conv = groups["conventional_cei_matched"]
    fixed = groups["forced_b50"]
    assert len(auto) == 21
    for r in auto:
        assert r["stopped_naturally"] == "True"
        assert int(r["n_model_building"]) >= 17
        assert int(r["n_confirmation"]) == 10
        assert float(r["stop_rho"]) <= .01
        assert int(r["total_evaluations"]) == int(r["n_model_building"]) + 10
    for a, c in zip(auto, conv):
        assert a["total_evaluations"] == c["total_evaluations"]
    a = np.array([float(r["best_R_ohm"]) for r in auto])
    c = np.array([float(r["best_R_ohm"]) for r in conv])
    f = np.array([float(r["best_R_ohm"]) for r in fixed])
    delta = a - c
    calls = np.array([int(r["total_evaluations"]) for r in auto])
    changed = []
    if old:
        for r in auto:
            previous = next(o for o in old if o["policy"] == r["policy"]
                            and o["target_M_nH"] == r["target_M_nH"])
            if r["total_evaluations"] != previous["total_evaluations"]:
                changed.append({"before": previous, "after": r})
    result = {
        "automatic_mean_calls": float(calls.mean()),
        "automatic_mean_R": float(a.mean()),
        "conventional_mean_R": float(c.mean()),
        "fixed50_mean_R": float(f.mean()),
        "savings_percent": float(100 * (1 - calls.mean()/50)),
        "resistance_penalty_percent": float(100 * (a.mean()/f.mean() - 1)),
        "wins_ties_losses": [int((delta < -1e-10).sum()),
                             int((np.abs(delta) <= 1e-10).sum()),
                             int((delta > 1e-10).sum())],
        "wilcoxon_p": float(wilcoxon(delta).pvalue),
        "feasible_counts": {p: sum(r["qualified"] == "True" for r in g)
                            for p, g in groups.items()},
        "changed_automatic_runs": changed,
    }
    (folder / "uncapped_comparison_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
