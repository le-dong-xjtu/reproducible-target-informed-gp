from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

root = Path(__file__).parents[2]
c = json.loads((root / "config" / "coil_problem.json").read_text())
s = json.loads((root / "config" / "surrogates" / "global_dual_gp.json").read_text())
w, g = c["wire"], c["geometry"]
pitch = w["diameter_mm"] * w["winding_factor"]
target = float(s["target_mH"])
epsilon = float(s["relative_tolerance"])
alpha = float(s["alpha"])
if not 0 <= alpha <= 1:
    raise ValueError("alpha must lie in [0, 1]")

pts = [(T, n) for T in range(g["layers"]["min"], g["layers"]["max"] + 1)
       for n in range(g["turns_per_layer"]["min"], g["turns_per_layer"]["max"] + 1)]
X = np.array([
    [(T - g["layers"]["min"]) / (g["layers"]["max"] - g["layers"]["min"]),
     (n - g["turns_per_layer"]["min"]) /
     (g["turns_per_layer"]["max"] - g["turns_per_layer"]["min"])]
    for T, n in pts
])


def ev(index):
    T, n = pts[index]
    N, H, thickness = T * n, n * pitch, T * pitch
    diameter = g["inner_diameter_mm"] + thickness
    L = .08 * N**2 * diameter**2 * 1e-4 / (3 * diameter + 9 * H + 10 * thickness)
    R = 4 * w["resistivity_ohm_mm"] * N * diameter / w["diameter_mm"]**2
    return {"point": [T, n], "L_mH": L, "R_ohm": R,
            "feasible": abs(L - target) / target <= epsilon}


def fit_gp(ids, field, restarts):
    values = np.array([ev(index)[field] for index in ids])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1., (1e-3, 1e3))
            * RBF([.25, .25], length_scale_bounds=(1e-2, 10.0))
            + WhiteKernel(1e-6, noise_level_bounds=(1e-8, 1e-2)),
            alpha=1e-10, normalize_y=True, random_state=s["seed"],
            n_restarts_optimizer=restarts,
        ).fit(X[ids], values)
        mean, std = model.predict(X, return_std=True)
    return mean, std


rng = random.Random(int(s["seed"]))
seen = set(rng.sample(range(len(pts)), int(s["initial_samples"])))
initial = sorted(seen)
initial_L = np.array([ev(index)["L_mH"] for index in initial])
initial_R = np.array([ev(index)["R_ohm"] for index in initial])
scale_L = max(float(np.std(initial_L)), 1e-12)
scale_R = max(float(np.std(initial_R)), 1e-12)
acquisition_trace = []

while len(seen) < int(s["maximum_true_evaluations"]):
    ids = sorted(seen)
    mean, std = fit_gp(ids, "L_mH", 3)
    mean_R, std_R = fit_gp(ids, "R_ohm", 3)
    unseen = np.array([index for index in range(len(pts)) if index not in seen])
    acquisition = ((1 - alpha) * std[unseen] / scale_L
                   + alpha * std_R[unseen] / scale_R)
    selected = int(unseen[int(np.argmax(acquisition))])
    acquisition_trace.append({"point": list(pts[selected]),
                              "score": float(np.max(acquisition))})
    seen.add(selected)

ids = sorted(seen)
mean, std = fit_gp(ids, "L_mH", 5)
mean_R, std_R = fit_gp(ids, "R_ohm", 5)
unseen = np.array([index for index in range(len(pts)) if index not in seen])
sigma_L = np.maximum(std[unseen], 1e-12)
lower, upper = target * (1 - epsilon), target * (1 + epsilon)
p_feasible = ndtr((upper - mean[unseen]) / sigma_L) - ndtr((lower - mean[unseen]) / sigma_L)
resistance_span = np.ptp(mean_R[unseen])
scaled_R = ((mean_R[unseen] - np.min(mean_R[unseen])) / resistance_span
            if resistance_span > 0 else np.zeros(len(unseen)))
validation_score = p_feasible * np.exp(-scaled_R)
final_ids = unseen[np.argsort(-validation_score)[:int(s["final_validation_count"])]]
final = [dict(ev(int(index)), predicted_mean_L_mH=float(mean[index]),
              sigma_L_mH=float(std[index]), predicted_mean_R_ohm=float(mean_R[index]),
              sigma_R_ohm=float(std_R[index])) for index in final_ids]
valid = [point for point in final if point["feasible"]]
decision = min(valid, key=lambda point: point["R_ohm"]) if valid else min(
    final, key=lambda point: abs(point["L_mH"] - target) / target
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
