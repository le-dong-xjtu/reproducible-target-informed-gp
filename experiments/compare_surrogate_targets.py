from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def truth_grid(config: dict) -> list[dict]:
    wire = config["wire"]
    geometry = config["geometry"]
    d = float(wire["diameter_mm"])
    p = d * float(wire["winding_factor"])
    rho = float(wire["resistivity_ohm_mm"])
    d0 = float(geometry["inner_diameter_mm"])
    grid = []
    for layers in range(int(geometry["layers"]["min"]), int(geometry["layers"]["max"]) + 1):
        for turns in range(int(geometry["turns_per_layer"]["min"]), int(geometry["turns_per_layer"]["max"]) + 1):
            total_turns = layers * turns
            height = turns * p
            thickness = layers * p
            mean_d = d0 + thickness
            inductance = (8e-6 * (total_turns ** 2) * (mean_d ** 2)) / (3.0 * mean_d + 9.0 * height + 10.0 * thickness)
            resistance = (4.0 * rho * total_turns * mean_d) / (d ** 2)
            grid.append({
                "T": layers,
                "n": turns,
                "N": total_turns,
                "L_mH": inductance,
                "R_ohm": resistance,
            })
    return grid


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
