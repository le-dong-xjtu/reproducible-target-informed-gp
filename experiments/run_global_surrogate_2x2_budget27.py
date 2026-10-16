"""Matched 2x2 global-surrogate benchmark at B=27.

Factors:
  model family: Gaussian process or ten-network MLP ensemble;
  model-building design: 17 random samples or 6 random + 11 uncertainty samples.

Every variant models both inductance and resistance, uses alpha=0.5 for active
selection, and applies the same 10-point resistance-aware verification rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from compare_surrogate_targets import ROOT, truth_grid, write_json

SEEDS = tuple(range(20261015, 20261025))
TARGETS = tuple(range(1, 88))
EPSILON = 0.025
BUDGET = 27
MODEL_EVALUATIONS = 17
INITIAL_EVALUATIONS = 6
VALIDATIONS = 10
ENSEMBLE_SIZE = 10
ALPHA = 0.5
VARIANTS = ("gp_random", "gp_active", "ensemble_random", "ensemble_active")
FIELDS = ("method", "budget", "seed", "target_mH", "qualified",
          "relative_resistance_regret", "bounded_loss_u", "decision_T",
          "decision_n", "decision_L_mH", "decision_R_ohm")


def unit_scale(values):
    values = np.asarray(values, dtype=float)
    span = np.ptp(values)
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def feasible(point, target):
    return abs(point["L_mH"] - target) / target <= EPSILON


def normalized_inputs(config):
    geometry = config["geometry"]
    return np.array([
        [(point["T"] - geometry["layers"]["min"]) /
         (geometry["layers"]["max"] - geometry["layers"]["min"]),
         (point["n"] - geometry["turns_per_layer"]["min"]) /
         (geometry["turns_per_layer"]["max"] - geometry["turns_per_layer"]["min"])]
        for point in truth_grid(config)
    ])


def fit_gp(x, y, ids, seed, restarts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1., (1e-3, 1e3))
            * RBF([.25, .25], length_scale_bounds=(1e-2, 10.0))
            + WhiteKernel(1e-6, noise_level_bounds=(1e-8, 1e-2)),
            alpha=1e-10, normalize_y=True, random_state=seed,
            n_restarts_optimizer=restarts,
        ).fit(x[ids], y[ids])
        return model.predict(x, return_std=True)


def fit_ensemble(x, y, ids, seed, response_offset):
    x_scaler = StandardScaler().fit(x[ids])
    y_scaler = StandardScaler().fit(y[ids, None])
    train_x = x_scaler.transform(x[ids])
    train_y = y_scaler.transform(y[ids, None]).ravel()
    query_x = x_scaler.transform(x)
    predictions = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for member in range(ENSEMBLE_SIZE):
            model = MLPRegressor(
                hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
                learning_rate_init=.01, alpha=.001, max_iter=1500,
                random_state=seed + response_offset + member,
            ).fit(train_x, train_y)
            prediction = y_scaler.inverse_transform(
                model.predict(query_x).reshape(-1, 1)
            ).ravel()
            predictions.append(prediction)
    predictions = np.asarray(predictions)
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("non-finite neural-ensemble prediction")
    return np.mean(predictions, axis=0), np.maximum(np.std(predictions, axis=0), 1e-12)


def fit_pair(family, x, inductance, resistance, ids, seed, final=False):
    if family == "gp":
        restarts = 5 if final else 3
        mean_l, std_l = fit_gp(x, inductance, ids, seed, restarts)
        mean_r, std_r = fit_gp(x, resistance, ids, seed + 100, restarts)
    else:
        mean_l, std_l = fit_ensemble(x, inductance, ids, seed, 1000)
        mean_r, std_r = fit_ensemble(x, resistance, ids, seed, 2000)
    return mean_l, std_l, mean_r, std_r


def build_model(variant, seed, x, inductance, resistance):
    family, policy = variant.split("_")
    rng = random.Random(seed)
    if policy == "random":
        seen = set(rng.sample(range(len(x)), MODEL_EVALUATIONS))
    else:
        seen = set(rng.sample(range(len(x)), INITIAL_EVALUATIONS))
        initial = sorted(seen)
        scale_l = max(float(np.std(inductance[initial])), 1e-12)
        scale_r = max(float(np.std(resistance[initial])), 1e-12)
        while len(seen) < MODEL_EVALUATIONS:
            ids = sorted(seen)
            _, std_l, _, std_r = fit_pair(
                family, x, inductance, resistance, ids, seed
            )
            unseen = np.array([index for index in range(len(x)) if index not in seen])
            acquisition = ((1 - ALPHA) * std_l[unseen] / scale_l
                           + ALPHA * std_r[unseen] / scale_r)
            seen.add(int(unseen[int(np.argmax(acquisition))]))
    ids = sorted(seen)
    predictions = fit_pair(family, x, inductance, resistance, ids, seed, final=True)
    return seen, predictions


def run_variant(task):
    variant, seed = task
    config = json.loads((ROOT / "config" / "coil_problem.json").read_text())
    grid = truth_grid(config)
    x = normalized_inputs(config)
    inductance = np.asarray([point["L_mH"] for point in grid])
    resistance = np.asarray([point["R_ohm"] for point in grid])
    seen, (mean_l, std_l, mean_r, _) = build_model(
        variant, seed, x, inductance, resistance
    )
    unseen = np.array([index for index in range(len(grid)) if index not in seen])
    sigma_l = np.maximum(std_l[unseen], 1e-12)
    scaled_r = unit_scale(mean_r[unseen])
    rows = []
    for target in TARGETS:
        lower, upper = target * (1 - EPSILON), target * (1 + EPSILON)
        p_feasible = (ndtr((upper - mean_l[unseen]) / sigma_l)
                      - ndtr((lower - mean_l[unseen]) / sigma_l))
        validation_score = p_feasible * np.exp(-scaled_r)
        final_ids = unseen[np.argsort(-validation_score)[:VALIDATIONS]]
        candidates = [grid[int(index)] for index in final_ids]
        valid = [point for point in candidates if feasible(point, target)]
        decision = min(valid, key=lambda point: point["R_ohm"]) if valid else min(
            candidates, key=lambda point: abs(point["L_mH"] - target) / target
        )
        optimum = min((point for point in grid if feasible(point, target)),
                      key=lambda point: point["R_ohm"])
        qualified = feasible(decision, target)
        regret = decision["R_ohm"] / optimum["R_ohm"] - 1 if qualified else None
        loss = regret / (1 + regret) if qualified else 1.0
        rows.append({
            "method": variant, "budget": BUDGET, "seed": seed,
            "target_mH": target, "qualified": qualified,
            "relative_resistance_regret": regret, "bounded_loss_u": loss,
            "decision_T": decision["T"], "decision_n": decision["n"],
            "decision_L_mH": decision["L_mH"], "decision_R_ohm": decision["R_ohm"],
        })
    return rows


def metrics(rows):
    losses = [float(row["bounded_loss_u"]) for row in rows]
    by_seed, by_target = defaultdict(list), defaultdict(list)
    for row in rows:
        value = float(row["bounded_loss_u"])
        by_seed[int(row["seed"])].append(value)
        by_target[int(float(row["target_mH"]))].append(value)
    seed_means = {seed: float(np.mean(values)) for seed, values in by_seed.items()}
    trimmed = [float(np.mean(sorted(by_target[target])[1:-1])) for target in TARGETS]
    return {
        "cases": len(rows),
        "qualification_rate": sum(row["qualified"].lower() == "true" for row in rows) / len(rows),
        "exact_optimum_cases": sum(value <= 1e-12 for value in losses),
        "exact_optimum_rate": sum(value <= 1e-12 for value in losses) / len(losses),
        "all_case_mean_u": float(np.mean(losses)),
        "targetwise_trimmed_mean_u": float(np.mean(trimmed)),
        "worst_seed_mean_u": max(seed_means.values()),
        "worst_seed": max(seed_means, key=seed_means.get),
    }


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = (ROOT / "results" / "coil_2d_relative_tolerance" /
              "surrogate_family_comparison_budget27")
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results.csv"
    existing = read_rows(results)
    completed = {(row["method"], int(row["seed"])) for row in existing}
    tasks = [(variant, seed) for variant in VARIANTS for seed in SEEDS
             if (variant, seed) not in completed]
    new_file = not results.exists()
    with results.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_variant, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                rows = future.result()
                writer.writerows(rows)
                stream.flush()
                print(f"checkpoint {index}/{len(tasks)} method={rows[0]['method']} "
                      f"seed={rows[0]['seed']}", flush=True)
    all_rows = read_rows(results)
    summaries = {variant: metrics([row for row in all_rows if row["method"] == variant])
                 for variant in VARIANTS}
    write_json(output / "summary.json", summaries)
    write_json(output / "manifest.json", {
        "status": "complete", "budget": BUDGET, "seeds": list(SEEDS),
        "targets_mH": list(TARGETS), "variants": list(VARIANTS),
        "ensemble_size": ENSEMBLE_SIZE, "global_response_alpha": ALPHA,
        "budget_composition": "17 reusable model-building evaluations + 10 final verifications per target",
        "random_policy": "17 seeded uniform random samples",
        "active_policy": "6 seeded random samples + 11 balanced scaled L/R uncertainty acquisitions",
        "validation": "same probability-of-feasibility times scaled resistance score for all variants",
        "decision_pool": "10 target-specific final model evaluations only",
    })
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
