import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_center_start_control as base

N_CONFIRM = 10
PF_WEIGHT = 0.75
DELTA_STOP = 0.01
MIN_MODEL_CALLS = 5
PATIENCE = 3
TARGETS = tuple(range(1, 88))
SEEDS = base.SEEDS
OUT = base.ROOT / "essay" / "data" / "random6_adaptive_stopping_uncapped_rbf"
ROWS = OUT / "results.csv"
FIELDS = (
    "seed",
    "target_mH",
    "qualified",
    "exact",
    "bounded_loss_u",
    "relative_resistance_regret",
    "decision_T",
    "decision_n",
    "decision_L_mH",
    "decision_R_ohm",
    "decision_stage",
    "oracle_calls",
    "n_model_building",
    "n_confirmation",
    "stopped_naturally",
    "first_feasible_evaluation",
    "final_rho",
    "model_building_ids",
    "confirmation_ids",
)


def p_feasible(target, mean_l, std_l, unseen):
    sigma = np.maximum(std_l[unseen], 1e-12)
    lower, upper = target * (1.0 - base.EPSILON), target * (1.0 + base.EPSILON)
    return ndtr((upper - mean_l[unseen]) / sigma) - ndtr(
        (lower - mean_l[unseen]) / sigma
    )


def run_one(task):
    seed, target = task
    selected = random.Random(seed).sample(range(len(base.POINTS)), 6)
    selected_set = set(selected)
    first_feasible = next(
        (
            position
            for position, index in enumerate(selected, 1)
            if base.is_feasible(index, target)
        ),
        None,
    )
    stop_streak = 0
    final_rho = None
    stopped = False

    while True:
        if len(selected) >= len(base.POINTS) - N_CONFIRM:
            raise RuntimeError("Candidate domain exhausted before automatic stopping")
        ids = np.asarray(selected, int)
        mean_l, std_l = base.fit(base.L_ALL, ids, seed, deterministic_starts=5)
        mean_r, std_r = base.fit(base.R_ALL, ids, seed, deterministic_starts=5)
        unseen = np.asarray(
            [i for i in range(len(base.POINTS)) if i not in selected_set], int
        )
        pf = p_feasible(target, mean_l, std_l, unseen)
        feasible_ids = [i for i in selected if base.is_feasible(i, target)]
        if not feasible_ids:
            score = PF_WEIGHT * base.unit(pf) + (1 - PF_WEIGHT) * base.unit(
                std_l[unseen]
            )
            final_rho = None
            stop_streak = 0
        else:
            incumbent = min(base.R_ALL[i] for i in feasible_ids)
            sigma_r = np.maximum(std_r[unseen], 1e-12)
            improvement = incumbent - mean_r[unseen]
            z = improvement / sigma_r
            ei = improvement * ndtr(z) + sigma_r * np.exp(-0.5 * z * z) / np.sqrt(
                2 * np.pi
            )
            score = pf * np.maximum(ei, 0) / incumbent
            final_rho = float(np.max(score))
            stop_streak = stop_streak + 1 if final_rho <= DELTA_STOP else 0
            if len(selected) >= MIN_MODEL_CALLS and stop_streak >= PATIENCE:
                stopped = True
                break
        chosen = int(unseen[int(np.argmax(score))])
        selected.append(chosen)
        selected_set.add(chosen)
        if first_feasible is None and base.is_feasible(chosen, target):
            first_feasible = len(selected)

    ids = np.asarray(selected, int)
    mean_l, std_l = base.fit(
        base.L_ALL, ids, seed, final=True, deterministic_starts=5
    )
    mean_r, _ = base.fit(
        base.R_ALL, ids, seed, final=True, deterministic_starts=5
    )
    unseen = np.asarray(
        [i for i in range(len(base.POINTS)) if i not in selected_set], int
    )
    pf = p_feasible(target, mean_l, std_l, unseen)
    confirmation_score = pf * np.exp(-base.unit(mean_r[unseen]))
    confirmation = [
        int(i)
        for i in unseen[
            np.argsort(-confirmation_score, kind="stable")[:N_CONFIRM]
        ]
    ]
    pool = selected + confirmation
    feasible_pool = [i for i in pool if base.is_feasible(i, target)]
    qualified = bool(feasible_pool)
    domain = np.flatnonzero(np.abs(base.L_ALL - target) / target <= base.EPSILON)
    optimum = int(domain[np.argmin(base.R_ALL[domain])])
    decision = (
        min(feasible_pool, key=lambda i: base.R_ALL[i])
        if qualified
        else min(pool, key=lambda i: abs(base.L_ALL[i] - target) / target)
    )
    regret = (
        float(base.R_ALL[decision] / base.R_ALL[optimum] - 1) if qualified else None
    )
    loss = regret / (1 + regret) if qualified else 1.0
    layers, turns = base.POINTS[decision]
    return {
        "seed": seed,
        "target_mH": target,
        "qualified": qualified,
        "exact": bool(qualified and regret <= 1e-12),
        "bounded_loss_u": loss,
        "relative_resistance_regret": regret,
        "decision_T": layers,
        "decision_n": turns,
        "decision_L_mH": float(base.L_ALL[decision]),
        "decision_R_ohm": float(base.R_ALL[decision]),
        "decision_stage": (
            "model_building" if decision in selected_set else "confirmation"
        ),
        "oracle_calls": len(pool),
        "n_model_building": len(selected),
        "n_confirmation": N_CONFIRM,
        "stopped_naturally": stopped,
        "first_feasible_evaluation": first_feasible,
        "final_rho": final_rho,
        "model_building_ids": json.dumps(selected),
        "confirmation_ids": json.dumps(confirmation),
    }


def read(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def deterministic_reference():
    development = read(
        base.ROOT / "essay/data/center_stopping_1pct_deterministic_gp_k5/results.csv"
    )
    heldout = read(base.ROOT / "essay/data/center_stopping_heldout_73/results.csv")
    reference = {}
    for row in development:
        reference.setdefault(int(row["target_mH"]), row)
    for row in heldout:
        reference[int(row["target_mH"])] = row
    assert set(reference) == set(TARGETS)
    return reference


def summarize(data):
    reference = deterministic_reference()
    losses = np.asarray([float(row["bounded_loss_u"]) for row in data])
    calls = np.asarray([int(row["oracle_calls"]) for row in data])
    exact = np.asarray([row["exact"].lower() == "true" for row in data])
    deterministic_losses = np.asarray(
        [float(reference[int(row["target_mH"])]["bounded_loss_u"]) for row in data]
    )
    grouped = defaultdict(list)
    for row in data:
        grouped[int(row["target_mH"])].append(row)
    variable_decisions = sum(
        len({(row["decision_T"], row["decision_n"]) for row in group}) > 1
        for group in grouped.values()
    )
    variable_losses = sum(
        len({row["bounded_loss_u"] for row in group}) > 1
        for group in grouped.values()
    )
    deterministic_calls = np.asarray(
        [int(reference[target]["oracle_calls"]) for target in TARGETS]
    )
    matched_deterministic_calls = np.asarray(
        [int(reference[int(row["target_mH"])]["oracle_calls"]) for row in data]
    )
    failure_targets = sorted(
        {int(row["target_mH"]) for row in data if row["exact"].lower() != "true"}
    )
    variable_decision_targets = sorted(
        target
        for target, group in grouped.items()
        if len({(row["decision_T"], row["decision_n"]) for row in group}) > 1
    )
    variable_loss_targets = sorted(
        target
        for target, group in grouped.items()
        if len({row["bounded_loss_u"] for row in group}) > 1
    )
    rng = np.random.default_rng(20261013)
    loss_matrix = np.asarray(
        [
            [
                float(row["bounded_loss_u"])
                for row in sorted(grouped[target], key=lambda item: int(item["seed"]))
            ]
            for target in TARGETS
        ]
    )
    call_matrix = np.asarray(
        [
            [
                int(row["oracle_calls"]) - int(reference[target]["oracle_calls"])
                for row in sorted(grouped[target], key=lambda item: int(item["seed"]))
            ]
            for target in TARGETS
        ]
    )
    draws = rng.integers(0, len(TARGETS), size=(100000, len(TARGETS)))
    boot_loss = loss_matrix[draws].mean(axis=(1, 2))
    boot_calls = call_matrix[draws].mean(axis=(1, 2))
    return {
        "random6": {
            "cases": len(data),
            "qualified_cases": int(
                sum(row["qualified"].lower() == "true" for row in data)
            ),
            "exact_cases": int(exact.sum()),
            "mean_bounded_loss_u": float(losses.mean()),
            "mean_calls": float(calls.mean()),
            "median_calls": float(np.median(calls)),
            "call_range": [int(calls.min()), int(calls.max())],
            "targets_with_seed_variable_decision": variable_decisions,
            "targets_with_seed_variable_loss": variable_losses,
            "variable_decision_targets_mH": variable_decision_targets,
            "variable_loss_targets_mH": variable_loss_targets,
            "nonexact_targets_mH": failure_targets,
            "loss_quantiles_95_99_max": np.quantile(
                losses, [0.95, 0.99, 1]
            ).tolist(),
        },
        "deterministic_center": {
            "unique_target_trajectories": 87,
            "qualified_targets": 87,
            "exact_targets": 87,
            "mean_bounded_loss_u": 0.0,
            "mean_calls": float(deterministic_calls.mean()),
            "median_calls": float(np.median(deterministic_calls)),
            "call_range": [
                int(deterministic_calls.min()),
                int(deterministic_calls.max()),
            ],
            "implementation_seed_variance": 0.0,
        },
        "paired_random_cases_vs_target_matched_deterministic": {
            "deterministic_wins": int(np.sum(losses > deterministic_losses)),
            "ties": int(np.sum(losses == deterministic_losses)),
            "deterministic_losses": int(np.sum(losses < deterministic_losses)),
            "random_minus_deterministic_mean_calls": float(
                np.mean(calls - matched_deterministic_calls)
            ),
            "target_cluster_bootstrap_95pct_ci_mean_loss_difference": np.quantile(
                boot_loss, [0.025, 0.975]
            ).tolist(),
            "target_cluster_bootstrap_95pct_ci_call_difference": np.quantile(
                boot_calls, [0.025, 0.975]
            ).tolist(),
        },
        "interpretation": (
            "Operational comparison under common acquisition, GP fitting, stopping, confirmation, "
            "without an evaluation cap. Different initial-set sizes also change active-call allocation; "
            "therefore this does not isolate determinism alone."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    existing = read(ROWS)
    done = {(int(row["seed"]), int(row["target_mH"])) for row in existing}
    tasks = [
        (seed, target)
        for seed in SEEDS
        for target in TARGETS
        if (seed, target) not in done
    ]
    new_file = not ROWS.exists()
    with ROWS.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, task): task for task in tasks}
            for count, future in enumerate(as_completed(futures), 1):
                writer.writerow(future.result())
                stream.flush()
                if count % 50 == 0 or count == len(tasks):
                    print(f"checkpoint {count}/{len(tasks)}", flush=True)
    data = read(ROWS)
    if len(data) != len(SEEDS) * len(TARGETS):
        raise RuntimeError(f"incomplete results: {len(data)}")
    summary = summarize(data)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "seeds": list(SEEDS),
                "targets_mH": list(TARGETS),
                "random_initial_count": 6,
                "deterministic_initial_count": 1,
                "pre_feasibility_score": "0.75*unit(pF)+0.25*unit(sigma_L)",
                "gp_optimizer": "deterministic Halton multistart K=5",
                "stop": {
                    "relative_cEI": 0.01,
                    "patience": 3,
                    "minimum_model_building": 5,
                },
                "budget": {
                    "maximum_model_building": None,
                    "confirmation": 10,
                    "maximum_total": None,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
