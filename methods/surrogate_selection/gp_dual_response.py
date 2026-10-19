import json
import random
import warnings
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

root = Path(__file__).resolve().parents[2]
config = json.loads((root / "config" / "coil_problem.json").read_text())
settings = json.loads((root / "config" / "surrogates" / "global_dual_gp.json").read_text())
wire, geometry = config["wire"], config["geometry"]
pitch = wire["diameter_mm"] * wire["winding_factor"]
target = float(settings["target_mH"])
epsilon = float(settings["relative_tolerance"])
alpha = float(settings["alpha"])
if not 0 <= alpha <= 1:
    raise ValueError("alpha must lie in [0, 1]")

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
        predicted_mean, predicted_std = model.predict(X, return_std=True)
    return predicted_mean, predicted_std


rng = random.Random(int(settings["seed"]))
seen = set(rng.sample(range(len(points)), int(settings["initial_samples"])))
initial = sorted(seen)
initial_L = np.array([evaluate_design(index)["L_mH"] for index in initial])
initial_R = np.array([evaluate_design(index)["R_ohm"] for index in initial])
scale_L = max(float(np.std(initial_L)), 1e-12)
scale_R = max(float(np.std(initial_R)), 1e-12)
acquisition_trace = []

while len(seen) < int(settings["maximum_true_evaluations"]):
    ids = sorted(seen)
    mean, std = fit_gp(ids, "L_mH", 3)
    mean_R, std_R = fit_gp(ids, "R_ohm", 3)
    unseen = np.array([index for index in range(len(points)) if index not in seen])
    acquisition = (1 - alpha) * std[unseen] / scale_L + alpha * std_R[unseen] / scale_R
    selected = int(unseen[int(np.argmax(acquisition))])
    acquisition_trace.append(
        {"point": list(points[selected]), "score": float(np.max(acquisition))}
    )
    seen.add(selected)

ids = sorted(seen)
mean, std = fit_gp(ids, "L_mH", 5)
mean_R, std_R = fit_gp(ids, "R_ohm", 5)
unseen = np.array([index for index in range(len(points)) if index not in seen])
sigma_L = np.maximum(std[unseen], 1e-12)
lower, upper = target * (1 - epsilon), target * (1 + epsilon)
p_feasible = ndtr((upper - mean[unseen]) / sigma_L) - ndtr(
    (lower - mean[unseen]) / sigma_L
)
resistance_span = np.ptp(mean_R[unseen])
scaled_R = (
    (mean_R[unseen] - np.min(mean_R[unseen])) / resistance_span
    if resistance_span > 0
    else np.zeros(len(unseen))
)
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
valid = [point for point in final if point["feasible"]]
decision = (
    min(valid, key=lambda point: point["R_ohm"])
    if valid
    else min(final, key=lambda point: abs(point["L_mH"] - target) / target)
)
out = {
    "method": "Global dual-response GP active learning",
    "alpha": alpha,
    "fixed_scales": {"L_mH": scale_L, "R_ohm": scale_R},
    "model_building_evaluations": len(seen),
    "final_validation_count": len(final),
    "decision": decision,
    "acquisition_trace": acquisition_trace,
}
output = root / "results" / "coil_2d_relative_tolerance" / "global_dual_gp_smoke_test"
output.mkdir(parents=True, exist_ok=True)
(output / "global_dual_gp_result.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
