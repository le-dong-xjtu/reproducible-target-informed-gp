import csv
import json
import random
from pathlib import Path

import numpy as np
import run_center_start_control as base

DATA = Path(__file__).resolve().parents[1] / "data"


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def check(row):
    target = float(row["target_mH"])
    selected = json.loads(row["model_building_ids"])
    confirmation = json.loads(row["confirmation_ids"])
    assert len(confirmation) == 10
    assert len(set(selected + confirmation)) == len(selected) + 10
    assert int(row["oracle_calls"]) == len(selected) + 10
    assert len(selected) >= 5
    assert row["stopped_naturally"].lower() == "true"
    assert float(row["final_rho"]) <= .01
    feasible = [i for i in selected + confirmation if base.is_feasible(i, target)]
    best = min(feasible, key=lambda i: base.R_ALL[i])
    assert np.isclose(float(row["decision_R_ohm"]), base.R_ALL[best])
    assert np.isclose(float(row["decision_L_mH"]), base.L_ALL[best])
    optimum = min(base.R_ALL[np.abs(base.L_ALL-target)/target <= .025])
    regret = base.R_ALL[best]/optimum - 1
    assert np.isclose(float(row["bounded_loss_u"]), regret/(1+regret), atol=1e-12)
    assert (row["exact"].lower() == "true") == (regret <= 1e-12)


def main():
    folder = DATA / "random6_adaptive_stopping_uncapped_rbf"
    random_rows = rows(folder / "results.csv")
    assert len(random_rows) == 870
    assert {(int(r["seed"]), int(r["target_mH"])) for r in random_rows} == {
        (s, t) for s in base.SEEDS for t in range(1, 88)}
    for r in random_rows:
        check(r)
        assert json.loads(r["model_building_ids"])[:6] == random.Random(int(r["seed"])).sample(range(1110), 6)
    assert sum(r["exact"].lower() == "true" for r in random_rows) == 859
    assert np.isclose(np.mean([int(r["oracle_calls"]) for r in random_rows]), 22.604597701149427)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["budget"]["maximum_model_building"] is None
    reference = {}
    for name in ("center_stopping_1pct_deterministic_gp_k5", "center_stopping_heldout_73"):
        for r in rows(DATA / name / "results.csv"):
            reference.setdefault(int(r["target_mH"]), r)
    assert len(reference) == 87
    for r in reference.values():
        check(r)
    assert all(float(r["bounded_loss_u"]) == 0 for r in reference.values())
    replay = rows(DATA / "uncapped_automatic_2d_rbf_replay" / "center.csv")
    assert len(replay) == 87
    for r in replay:
        old = reference[int(r["target_mH"])]
        assert int(r["oracle_calls"]) == int(old["oracle_calls"])
        assert json.loads(r["model_building_ids"]) == json.loads(old["model_building_ids"])
        assert json.loads(r["confirmation_ids"]) == json.loads(old["confirmation_ids"])
        history = json.loads(r["refit_history"])
        assert all(h["rho"] is not None and h["rho"] <= .01 for h in history[-3:])
    print("PASS: 870 uncapped RBF random-six runs, 859 exact; 87 uncapped center histories match original evidence.")


if __name__ == "__main__":
    main()
