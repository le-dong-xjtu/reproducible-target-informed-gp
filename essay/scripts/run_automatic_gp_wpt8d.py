import csv
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_center_start_control import deterministic_multistart
from wpt_partial_inductance_model import (
    PartialMutualInductanceCalculator,
    PlanarPolygonCoil,
)

LOW = np.array([4.0, 5.0, 1.0, 30.0, 4.0, 5.0, 1.0, 30.0], dtype=float)
HIGH = np.array([10.0, 15.0, 3.0, 80.0, 10.0, 15.0, 3.0, 80.0], dtype=float)
TARGET_M = 1500.0
EPSILON = 0.025
LOWER_M = TARGET_M * (1.0 - EPSILON)
UPPER_M = TARGET_M * (1.0 + EPSILON)
WIRE_DIAMETER_MM = 0.68
WIRE_AREA_MM2 = math.pi * (WIRE_DIAMETER_MM / 2.0) ** 2
CU_RESISTIVITY_OHM_MM = 1.851e-5
TRANSLATION_MM = (0.0, 0.0, 70.0)
ROTATION_RAD = (0.0, 0.0, 0.0)
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "essay" / "data" / "wpt_8d_automated_cei"
N_CANDIDATES = 2048
N_CONFIRM_CANDIDATES = 4096
N_CONFIRM = 10
MAX_MODEL_CALLS = 40
CONFIRM_POOL_START = 1 + 39 * N_CANDIDATES
STOP_DELTA = 0.01
STOP_PATIENCE = 3
MIN_MODEL_CALLS = 2 * len(LOW) + 1
FEASIBILITY_WEIGHT = 0.75
GP_STARTS = 5
CENTER = (LOW + HIGH) / 2
CENTER[[0, 1, 4, 5]] = np.rint(CENTER[[0, 1, 4, 5]])


def normalize(points):
    points = np.asarray(points, dtype=float)
    return (points - LOW) / (HIGH - LOW)


def unit(values):
    values = np.asarray(values, dtype=float)
    span = np.ptp(values)
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def evaluate(point):
    ns_t, nt_t, s_t, ri_t, ns_r, nt_r, s_r, ri_r = np.asarray(point, dtype=float)
    ns_t_i, nt_t_i = int(round(ns_t)), int(round(nt_t))
    ns_r_i, nt_r_i = int(round(ns_r)), int(round(nt_r))
    tx = PlanarPolygonCoil(
        ns=ns_t_i, nt=nt_t_i, s=float(s_t), r_i=float(ri_t), w=WIRE_DIAMETER_MM
    )
    rx = PlanarPolygonCoil(
        ns=ns_r_i, nt=nt_r_i, s=float(s_r), r_i=float(ri_r), w=WIRE_DIAMETER_MM
    )
    m_nh = PartialMutualInductanceCalculator.calculate_mutual_inductance(
        tx=tx, rx=rx, translation=TRANSLATION_MM, rotation=ROTATION_RAD
    )["M_nH"]
    r_ohm = 0.0
    for ns_i, nt_i, s_val, ri_val in (
        (ns_t_i, nt_t_i, float(s_t), float(ri_t)),
        (ns_r_i, nt_r_i, float(s_r), float(ri_r)),
    ):
        k = np.arange(nt_i, dtype=float)
        r_k = ri_val + k * (s_val + WIRE_DIAMETER_MM)
        r_ohm += (CU_RESISTIVITY_OHM_MM / WIRE_AREA_MM2) * float(
            np.sum(2.0 * ns_i * r_k * math.sin(math.pi / ns_i))
        )
    return np.array([float(m_nh), float(r_ohm)], dtype=float)


class SobolStream:
    def __init__(self):
        self.points = np.empty((0, 8))

    def __getitem__(self, key):
        needed = key.stop if isinstance(key, slice) else key + 1
        if needed is None:
            needed = max(len(self.points), 131072)
        if needed > len(self.points):
            power = int(math.ceil(math.log2(needed)))
            self.points = qmc.scale(
                qmc.Sobol(8, scramble=False).random_base2(power), LOW, HIGH
            )
        return self.points[key]


def sobol_stream():
    return SobolStream()


def canonical(points):
    points = np.asarray(points, float).copy()
    points[:, [0, 1, 4, 5]] = np.rint(points[:, [0, 1, 4, 5]])
    return points


def unseen_unique(points, observed):
    seen = {tuple(row) for row in np.asarray(observed)}
    answer = []
    local = set()
    for row in canonical(points):
        key = tuple(row)
        if key not in seen and key not in local:
            answer.append(row)
            local.add(key)
    return np.asarray(answer, float)


def gps(x, y):
    models = []
    optimize = len(x) >= 4
    for output in range(2):
        if optimize:
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(0.35, (0.03, 3.0), nu=2.5)
                + WhiteKernel(1e-5, (1e-8, 1e-2))
            )
            optimizer = deterministic_multistart(GP_STARTS)
        else:
            kernel = (
                ConstantKernel(1.0, constant_value_bounds="fixed")
                * Matern(0.35, length_scale_bounds="fixed", nu=2.5)
                + WhiteKernel(1e-5, noise_level_bounds="fixed")
            )
            optimizer = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            model = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                alpha=1e-6,
                optimizer=optimizer,
                n_restarts_optimizer=0,
            ).fit(x, y[:, output])
        models.append(model)
    return models


def components(models, candidates, observed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_m, sd_m = models[0].predict(candidates, return_std=True)
        mu_r, sd_r = models[1].predict(candidates, return_std=True)
    if not all(np.isfinite(value).all() for value in (mu_m, sd_m, mu_r, sd_r)):
        raise FloatingPointError("non-finite deterministic WPT GP prediction")
    sd_m = np.maximum(sd_m, 1e-12)
    sd_r = np.maximum(sd_r, 1e-12)
    pf = ndtr((UPPER_M - mu_m) / sd_m) - ndtr((LOWER_M - mu_m) / sd_m)
    feasible = observed[(observed[:, 0] >= LOWER_M) & (observed[:, 0] <= UPPER_M)]
    if len(feasible):
        incumbent = float(feasible[:, 1].min())
        improvement = incumbent - mu_r
        z = improvement / sd_r
        ei = (
            improvement * ndtr(z)
            + sd_r * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        )
        cei = pf * np.maximum(ei, 0) / incumbent
    else:
        incumbent, cei = None, None
    return pf, mu_r, sd_m, cei, incumbent


def main():
    started = time.monotonic()
    stream = sobol_stream()
    x_raw = np.asarray([CENTER], float)
    y = np.asarray([evaluate(CENTER)])
    trace = [
        {
            "model_call": 1,
            "stage": "center",
            "M_nH": y[0, 0],
            "R_ohm": y[0, 1],
            "rho": "",
            "below_threshold_streak": 0,
        }
    ]
    first_feasible = 1 if LOWER_M <= y[0, 0] <= UPPER_M else None
    streak = 0
    stopped_naturally = False
    final_rho = None

    while True:
        step = len(x_raw) - 1
        block = stream[1 + step * N_CANDIDATES : 1 + (step + 1) * N_CANDIDATES]
        candidates = unseen_unique(block, x_raw)
        models = gps(normalize(x_raw), y)
        pf, _, sd_m, cei, _ = components(models, normalize(candidates), y)
        if first_feasible is None:
            score = FEASIBILITY_WEIGHT * unit(pf) + (
                1 - FEASIBILITY_WEIGHT
            ) * unit(sd_m)
            rho, stage = None, "feasibility"
        else:
            rho = float(np.max(cei))
            final_rho = rho
            stage = "cei"
            streak = streak + 1 if rho <= STOP_DELTA else 0
            if len(x_raw) >= MIN_MODEL_CALLS and streak >= STOP_PATIENCE:
                stopped_naturally = True
                break
            score = cei
        chosen = candidates[int(np.argmax(score))]
        value = evaluate(chosen)
        x_raw = np.vstack([x_raw, chosen])
        y = np.vstack([y, value])
        if first_feasible is None and LOWER_M <= value[0] <= UPPER_M:
            first_feasible = len(x_raw)
        trace.append(
            {
                "model_call": len(x_raw),
                "stage": stage,
                "M_nH": value[0],
                "R_ohm": value[1],
                "rho": "" if rho is None else rho,
                "below_threshold_streak": streak,
            }
        )

    confirm_start = CONFIRM_POOL_START
    candidates = unseen_unique(
        stream[confirm_start : confirm_start + N_CONFIRM_CANDIDATES], x_raw
    )
    models = gps(normalize(x_raw), y)
    pf, mu_r, _, _, _ = components(models, normalize(candidates), y)
    confirmation_score = pf * np.exp(-unit(mu_r))
    chosen = np.argsort(-confirmation_score, kind="stable")[:N_CONFIRM]
    confirm_x = candidates[chosen]
    confirm_y = np.asarray([evaluate(point) for point in confirm_x])
    all_x, all_y = np.vstack([x_raw, confirm_x]), np.vstack([y, confirm_y])
    feasible = np.where((all_y[:, 0] >= LOWER_M) & (all_y[:, 0] <= UPPER_M))[0]
    qualified = bool(len(feasible))
    decision = (
        int(feasible[np.argmin(all_y[feasible, 1])])
        if qualified
        else int(np.argmin(np.abs(all_y[:, 0] - TARGET_M)))
    )
    result = {
        "qualified": qualified,
        "exact_reproducible": True,
        "best_M_nH": float(all_y[decision, 0]),
        "best_R_ohm": float(all_y[decision, 1]),
        "relative_target_error_pct": float(
            abs(all_y[decision, 0] - TARGET_M) / TARGET_M * 100
        ),
        "design": all_x[decision].tolist(),
        "decision_stage": (
            "model_building" if decision < len(x_raw) else "confirmation"
        ),
        "first_feasible_evaluation": first_feasible,
        "n_model_building": len(x_raw),
        "n_confirmation": N_CONFIRM,
        "total_evaluations": len(all_x),
        "stopped_naturally": stopped_naturally,
        "final_below_threshold_streak": streak,
        "final_rho": final_rho,
        "runtime_sec": time.monotonic() - started,
    }
    manifest = {
        "task": "minimize resistance at M=1500 nH within 2.5 percent",
        "dimensions": 8,
        "pose": "fixed nominal",
        "center": CENTER.tolist(),
        "pre_feasibility_weights": {"pF": 0.75, "sigma_M": 0.25},
        "acquisition_after_feasibility": "ordinary cEI",
        "stop": {
            "maximum_relative_cEI": STOP_DELTA,
            "patience": STOP_PATIENCE,
            "minimum_model_building": MIN_MODEL_CALLS,
        },
        "budget": {
            "maximum_model_building": None,
            "confirmation": N_CONFIRM,
            "maximum_total": None,
        },
        "gp": {
            "kernel": "isotropic Matern-5/2",
            "deterministic_starts": GP_STARTS,
            "fixed_prior_for_observations_below": 4,
        },
        "candidates": {
            "sequence": "unscrambled Sobol",
            "per_step": N_CANDIDATES,
            "confirmation_pool": N_CONFIRM_CANDIDATES,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (OUT / "trace.csv").open("w", newline="") as stream_out:
        writer = csv.DictWriter(stream_out, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    (OUT / "model_building_designs.json").write_text(
        json.dumps(x_raw.tolist(), indent=2) + "\n"
    )
    (OUT / "confirmation_designs.json").write_text(
        json.dumps(confirm_x.tolist(), indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
