import csv
import json

import numpy as np

import run_wpt_starting_policy_screen as experiment


def main():
    summary = json.loads((experiment.OUT / "summary.json").read_text())
    for result in summary["results"]:
        folder = experiment.OUT / result["policy"]
        with (folder / "trace.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == result["total_evaluations"]
        assert sum(row["stage"] == "verification" for row in rows) == 10
        assert result["n_model_building"] + 10 == len(rows)
        points = [tuple(json.loads(row["design"])) for row in rows]
        assert len(set(points)) == len(points)
        incumbent = None
        for row, point in zip(rows, points):
            mutual, resistance = experiment.auto.evaluate(np.asarray(point))
            assert np.isclose(mutual, float(row["M_nH"]), rtol=1e-10)
            assert np.isclose(resistance, float(row["R_ohm"]), rtol=1e-10)
            feasible = abs(mutual-experiment.TARGET) <= experiment.auto.EPSILON*experiment.TARGET
            assert feasible == (row["feasible"] == "True")
            if feasible:
                incumbent = resistance if incumbent is None else min(incumbent, resistance)
            if incumbent is None:
                assert row["best_feasible_R_ohm"] == ""
            else:
                assert np.isclose(incumbent, float(row["best_feasible_R_ohm"]))
        assert np.isclose(incumbent, result["best_R_ohm"])
        refits = json.loads((folder / "refits.json").read_text())
        assert result["stopped_naturally"], "Uncapped automatic run must terminate by the stopping rule"
        if result["stopped_naturally"]:
            assert result["n_model_building"] >= experiment.auto.MIN_MODEL_CALLS
            assert refits[-1]["observations"] == result["n_model_building"]
            assert all(r["rho"] is not None and r["rho"] <= experiment.auto.STOP_DELTA
                       for r in refits[-experiment.auto.STOP_PATIENCE:])
        else:
            assert result["n_model_building"] == experiment.auto.MAX_MODEL_CALLS
    random = summary["results"][1:]
    assert max(random, key=lambda r: r["best_R_ohm"])["policy"] == summary["selected_worst_policy"]
    print(f"PASS: {len(summary['results'])} trajectories; responses, budgets, uniqueness, incumbents, stopping, and worst-seed selection.")


if __name__ == "__main__":
    main()
