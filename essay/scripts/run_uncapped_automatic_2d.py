import argparse
import csv
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_center_start_control as base

DEV = (1, 2, 3, 4, 5, 11, 20, 30, 39, 49, 58, 68, 77, 87)
OUT = base.ROOT / "essay/data/uncapped_automatic_2d_rbf_replay"


def run(task):
    policy, target, seed = task
    selected = (
        [base.CENTER_ID]
        if policy == "center"
        else random.Random(seed).sample(range(len(base.POINTS)), 6)
    )
    initial = len(selected)
    first = next(
        (j + 1 for j, i in enumerate(selected) if base.is_feasible(i, target)), None
    )
    streak = 0
    history = []
    while True:
        ids = np.asarray(selected)
        ml, sl = base.fit(base.L_ALL, ids, seed, deterministic_starts=5)
        mr, sr = base.fit(base.R_ALL, ids, seed, deterministic_starts=5)
        unseen = np.setdiff1d(np.arange(len(base.POINTS)), ids)
        if len(unseen) < 10:
            raise RuntimeError("Candidate domain exhausted before automatic stopping")
        sl = np.maximum(sl[unseen], 1e-12)
        pf = ndtr((target * 1.025 - ml[unseen]) / sl) - ndtr(
            (target * 0.975 - ml[unseen]) / sl
        )
        feasible = [i for i in selected if base.is_feasible(i, target)]
        rho = None
        if feasible:
            incumbent = float(min(base.R_ALL[feasible]))
            sigma = np.maximum(sr[unseen], 1e-12)
            improvement = incumbent - mr[unseen]
            z = improvement / sigma
            score = (
                pf
                * np.maximum(
                    improvement * ndtr(z)
                    + sigma * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi),
                    0,
                )
                / incumbent
            )
            rho = float(max(score))
            streak = streak + 1 if rho <= 0.01 else 0
        else:
            score = 0.75 * base.unit(pf) + 0.25 * base.unit(sl)
            streak = 0
        history.append({"observations": len(selected), "rho": rho, "streak": streak})
        if len(selected) >= 5 and streak >= 3:
            break
        chosen = int(unseen[np.argmax(score)])
        selected.append(chosen)
        if first is None and base.is_feasible(chosen, target):
            first = len(selected)
    ranking = pf * np.exp(-base.unit(mr[unseen]))
    confirmation = unseen[np.argsort(-ranking, kind="stable")[:10]].tolist()
    feasible_all = [
        i for i in selected + confirmation if base.is_feasible(i, target)
    ]
    decision = min(feasible_all, key=lambda i: base.R_ALL[i])
    domain = np.flatnonzero(np.abs(base.L_ALL - target) / target <= 0.025)
    optimum = float(min(base.R_ALL[domain]))
    regret = float(base.R_ALL[decision] / optimum - 1)
    return {
        "policy": policy,
        "target_mH": target,
        "seed": seed,
        "split": "development" if target in DEV else "heldout",
        "qualified": True,
        "exact": regret <= 1e-12,
        "bounded_loss_u": regret / (1 + regret),
        "relative_resistance_regret": regret,
        "decision_T": base.POINTS[decision][0],
        "decision_n": base.POINTS[decision][1],
        "decision_L_mH": float(base.L_ALL[decision]),
        "decision_R_ohm": float(base.R_ALL[decision]),
        "oracle_calls": len(selected) + 10,
        "n_initial": initial,
        "n_model_building": len(selected),
        "n_confirmation": 10,
        "stopped_naturally": True,
        "first_feasible_evaluation": first,
        "final_rho": rho,
        "model_building_ids": json.dumps(selected),
        "confirmation_ids": json.dumps(confirmation),
        "incumbent_exact_at_stop": abs(incumbent - optimum) <= 1e-12,
        "refit_history": json.dumps(history),
    }


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--policy", choices=("center", "random6"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.policy}.csv"
    rows = read_rows(path)
    done = {(int(r["target_mH"]), int(r["seed"])) for r in rows}
    seeds = [20261015] if args.policy == "center" else base.SEEDS
    tasks = [
        (args.policy, t, s)
        for t in range(1, 88)
        for s in seeds
        if (t, s) not in done
    ]
    with path.open("a", newline="") as stream, ProcessPoolExecutor(
        args.workers
    ) as pool:
        writer = None
        for i, result in enumerate(
            as_completed([pool.submit(run, task) for task in tasks]), 1
        ):
            row = result.result()
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not rows:
                    writer.writeheader()
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            print(
                f"{args.policy} {i}/{len(tasks)} target={row['target_mH']} "
                f"seed={row['seed']} exact={row['exact']} B={row['oracle_calls']}",
                flush=True,
            )
    summary = {}
    for split in ("all", "development", "heldout"):
        group = [r for r in rows if split == "all" or r["split"] == split]
        calls = [int(r["oracle_calls"]) for r in group]
        summary[split] = {
            "cases": len(group),
            "exact": sum(float(r["bounded_loss_u"]) <= 1e-12 for r in group),
            "mean_calls": float(np.mean(calls)),
            "min_calls": min(calls),
            "max_calls": max(calls),
            "mean_loss": float(np.mean([float(r["bounded_loss_u"]) for r in group])),
        }
    (OUT / f"{args.policy}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
