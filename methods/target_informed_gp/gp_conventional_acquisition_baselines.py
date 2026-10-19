import json
import random
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

root = Path(__file__).resolve().parents[2]
config = json.loads((root / "config" / "coil_problem.json").read_text())
settings = json.loads(
    (root / "config" / "surrogates" / "target_conditioned_gp.json").read_text()
)
acquisition_mode = settings.get("acquisition_mode", "cei")
if acquisition_mode not in {"cei", "clcb", "ivr"}:
    raise ValueError("acquisition_mode must be 'cei', 'clcb', or 'ivr'")
wire, geometry = config["wire"], config["geometry"]
pitch = wire["diameter_mm"] * wire["winding_factor"]
target = float(settings["target_mH"])
epsilon = float(settings["relative_tolerance"])
points = [
    (layers, turns)
    for layers in range(geometry["layers"]["min"], geometry["layers"]["max"] + 1)
    for turns in range(
        geometry["turns_per_layer"]["min"], geometry["turns_per_layer"]["max"] + 1
    )
]
X = np.array([
    [
        (layers - geometry["layers"]["min"])
        / (geometry["layers"]["max"] - geometry["layers"]["min"]),
        (turns - geometry["turns_per_layer"]["min"])
        / (geometry["turns_per_layer"]["max"] - geometry["turns_per_layer"]["min"]),
    ]
    for layers, turns in points
])


def evaluate_design(index):
    layers, turns = points[index]
    total_turns = layers * turns
    height = turns * pitch
    thickness = layers * pitch
    diameter = geometry["inner_diameter_mm"] + thickness
    inductance = (
        0.08
        * total_turns**2
        * diameter**2
        * 1e-4
        / (3 * diameter + 9 * height + 10 * thickness)
    )
    resistance = (
        4 * wire["resistivity_ohm_mm"] * total_turns * diameter / wire["diameter_mm"] ** 2
    )
    return {
        "point": [layers, turns],
        "L_mH": inductance,
        "R_ohm": resistance,
        "feasible": abs(inductance - target) / target <= epsilon,
    }


def fit_gp(ids, field, restarts):
    values = np.array([evaluate_design(index)[field] for index in ids])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3))
            * RBF([0.25, 0.25], length_scale_bounds=(1e-2, 10.0))
            + WhiteKernel(1e-6, noise_level_bounds=(1e-8, 1e-2)),
            alpha=1e-10,
            normalize_y=True,
            random_state=settings["seed"],
            n_restarts_optimizer=restarts,
        ).fit(X[ids], values)
        predicted = model.predict(X, return_std=True)
    if not all(np.all(np.isfinite(array)) for array in predicted):
        raise FloatingPointError(f"Non-finite {field} GP prediction")
    return predicted[0], predicted[1], model


def ivr_score(model, train_ids, query_ids, roi_ids, roi_weights):
    x_train = X[train_ids]
    x_query = X[query_ids]
    x_roi = X[roi_ids]
    kernel = model.kernel_
    train_to_query = kernel(x_train, x_query)
    train_to_roi = kernel(x_train, x_roi)
    projected_query = solve_triangular(model.L_, train_to_query, lower=True)
    projected_roi = solve_triangular(model.L_, train_to_roi, lower=True)
    posterior_cross = kernel(x_roi, x_query) - projected_roi.T @ projected_query
    posterior_var_query = np.maximum(
        kernel.diag(x_query) - np.sum(projected_query**2, axis=0), 1e-12
    )
    return np.sum(roi_weights[:, None] * posterior_cross**2, axis=0) / posterior_var_query


def unit_scale(values):
    values = np.asarray(values, dtype=float)
    span = np.ptp(values)
    return (values - np.min(values)) / span if span > 0 else np.zeros_like(values)


rng = random.Random(settings["seed"])
seen = set(rng.sample(range(len(points)), int(settings["initial_samples"])))
initial_seen = set(seen)
initial_L = np.array([evaluate_design(index)["L_mH"] for index in sorted(initial_seen)])
initial_R = np.array([evaluate_design(index)["R_ohm"] for index in sorted(initial_seen)])
global_scale_L = max(float(np.std(initial_L)), 1e-12)
global_scale_R = max(float(np.std(initial_R)), 1e-12)
acquisition_trace = []
steps = int(settings["maximum_true_evaluations"]) - len(seen)

for step in range(steps):
    ids = sorted(seen)
    mean, std, model_L = fit_gp(ids, "L_mH", 3)
    mean_R, std_R, model_R = fit_gp(ids, "R_ohm", 3)
    unseen = np.array([index for index in range(len(points)) if index not in seen])
    sigma_L = np.maximum(std[unseen], 1e-12)
    lower, upper = target * (1 - epsilon), target * (1 + epsilon)
    p_feasible = ndtr((upper - mean[unseen]) / sigma_L) - ndtr(
        (lower - mean[unseen]) / sigma_L
    )
    feasible_seen = [
        evaluate_design(index) for index in ids if evaluate_design(index)["feasible"]
    ]
    if feasible_seen:
        incumbent = min(point["R_ohm"] for point in feasible_seen)
        sigma_R = np.maximum(std_R[unseen], 1e-12)
        z = (incumbent - mean_R[unseen]) / sigma_R
        expected_improvement = (incumbent - mean_R[unseen]) * ndtr(
            z
        ) + sigma_R * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        local = p_feasible * np.maximum(expected_improvement, 0) / incumbent
        phase = "resistance_improvement"
    else:
        local = p_feasible
        phase = "feasibility_discovery"
    if acquisition_mode == "cei":
        acquisition = local
    elif acquisition_mode == "clcb":
        kappa = float(settings.get("lcb_kappa", 2.0))
        resistance_lcb = mean_R[unseen] - kappa * np.maximum(std_R[unseen], 1e-12)
        acquisition = p_feasible * np.exp(-unit_scale(resistance_lcb))
        phase = "constrained_lcb"
    else:
        roi_size = min(int(settings.get("ivr_roi_size", 128)), len(points))
        global_fraction = float(settings.get("ivr_global_fraction", 0.25))
        global_count = max(1, int(round(global_fraction * roi_size)))
        local_count = roi_size - global_count
        global_ids = np.linspace(0, len(points) - 1, global_count, dtype=int)
        local_ids = np.argsort(
            -np.asarray(
                ndtr((upper - mean) / np.maximum(std, 1e-12))
                - ndtr((lower - mean) / np.maximum(std, 1e-12))
            )
        )[:local_count]
        roi = np.unique(np.concatenate([global_ids, local_ids]))
        full_p = np.asarray(
            ndtr((upper - mean[roi]) / np.maximum(std[roi], 1e-12))
            - ndtr((lower - mean[roi]) / np.maximum(std[roi], 1e-12))
        )
        weights = global_fraction * np.ones(len(roi)) / len(roi)
        weights += (1 - global_fraction) * (
            full_p / full_p.sum() if full_p.sum() > 0 else np.ones(len(roi)) / len(roi)
        )
        score_L = ivr_score(model_L, ids, unseen, roi, weights)
        score_R = ivr_score(model_R, ids, unseen, roi, weights)
        response_weight = float(settings.get("ivr_resistance_weight", 0.5))
        acquisition = (1 - response_weight) * unit_scale(
            score_L
        ) + response_weight * unit_scale(score_R)
        phase = "weighted_integrated_variance_reduction"

    local_weight = float(settings.get("local_weight", 1.0))
    global_weight = float(settings.get("global_weight", 0.0))
    if (
        local_weight < 0
        or global_weight < 0
        or not np.isclose(local_weight + global_weight, 1.0)
    ):
        raise ValueError("local_weight and global_weight must be nonnegative and sum to one")
    global_alpha = float(settings.get("global_response_alpha", 0.5))
    if not 0 <= global_alpha <= 1:
        raise ValueError("global_response_alpha must lie in [0, 1]")
    global_acquisition = (1 - global_alpha) * sigma_L / global_scale_L + global_alpha * np.maximum(
        std_R[unseen], 1e-12
    ) / global_scale_R
    acquisition = local_weight * unit_scale(acquisition) + global_weight * unit_scale(
        global_acquisition
    )
    selected = int(unseen[int(np.argmax(acquisition))])
    acquisition_trace.append(
        {
            "step": step + 1,
            "point": list(points[selected]),
            "phase": phase,
            "local_weight": local_weight,
            "score": float(np.max(acquisition)),
        }
    )
    seen.add(selected)

ids = sorted(seen)
mean, std, model_L = fit_gp(ids, "L_mH", 5)
mean_R, std_R, model_R = fit_gp(ids, "R_ohm", 5)
unseen = np.array([index for index in range(len(points)) if index not in seen])
sigma_L = np.maximum(std[unseen], 1e-12)
lower, upper = target * (1 - epsilon), target * (1 + epsilon)
p_feasible = ndtr((upper - mean[unseen]) / sigma_L) - ndtr(
    (lower - mean[unseen]) / sigma_L
)
scaled_R = unit_scale(mean_R[unseen])
validation_score = p_feasible * np.exp(-scaled_R)
final_ids = unseen[
    np.argsort(-validation_score)[: int(settings["final_validation_count"])]
]
final = [
    dict(
        evaluate_design(int(index)),
        predicted_mean_L_mH=float(mean[index]),
        sigma_L_mH=float(std[index]),
        predicted_mean_R_ohm=float(mean_R[index]),
        sigma_R_ohm=float(std_R[index]),
    )
    for index in final_ids
]
decision_pool = [
    dict(
        evaluate_design(index),
        source="initial" if index in initial_seen else "active_acquisition",
    )
    for index in ids
]
decision_pool += [dict(point, source="validation") for point in final]
valid = [point for point in decision_pool if point["feasible"]]
decision = (
    min(valid, key=lambda point: point["R_ohm"])
    if valid
    else min(decision_pool, key=lambda point: abs(point["L_mH"] - target) / target)
)
out = {
    "method": {
        "cei": "Conventional GP constrained expected improvement",
        "clcb": "Constrained GP lower confidence bound",
        "ivr": "Target-weighted integrated variance reduction",
    }[acquisition_mode],
    "acquisition_mode": acquisition_mode,
    "mixture_weights": {
        "local": float(settings.get("local_weight", 1.0)),
        "global": float(settings.get("global_weight", 0.0)),
        "global_response_alpha": float(settings.get("global_response_alpha", 0.5)),
    },
    "target_mH": target,
    "relative_tolerance": epsilon,
    "model_building_evaluations": len(seen),
    "final_validation_count": len(final),
    "decision": decision,
    "final_validation": final,
    "acquisition_trace": acquisition_trace,
}
output = (
    root
    / "results"
    / "coil_2d_relative_tolerance"
    / "target_conditioned_gp_smoke_test"
)
output.mkdir(parents=True, exist_ok=True)
(output / "target_conditioned_gp_result.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
print(json.dumps(out, indent=2))
