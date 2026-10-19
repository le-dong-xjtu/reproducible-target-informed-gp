import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def truth_grid(config):
    wire = config["wire"]
    geometry = config["geometry"]
    diameter = float(wire["diameter_mm"])
    pitch = diameter * float(wire["winding_factor"])
    resistivity = float(wire["resistivity_ohm_mm"])
    inner_diameter = float(geometry["inner_diameter_mm"])
    grid = []
    for layers in range(
        int(geometry["layers"]["min"]), int(geometry["layers"]["max"]) + 1
    ):
        for turns in range(
            int(geometry["turns_per_layer"]["min"]),
            int(geometry["turns_per_layer"]["max"]) + 1,
        ):
            total_turns = layers * turns
            height = turns * pitch
            thickness = layers * pitch
            mean_diameter = inner_diameter + thickness
            inductance = (
                8e-6 * (total_turns**2) * (mean_diameter**2)
            ) / (3.0 * mean_diameter + 9.0 * height + 10.0 * thickness)
            resistance = (4.0 * resistivity * total_turns * mean_diameter) / (
                diameter**2
            )
            grid.append(
                {
                    "T": layers,
                    "n": turns,
                    "N": total_turns,
                    "L_mH": inductance,
                    "R_ohm": resistance,
                }
            )
    return grid


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    config = json.loads((ROOT / "config" / "coil_problem.json").read_text())
    grid = truth_grid(config)
    inductance_values = [point["L_mH"] for point in grid]
    resistance_values = [point["R_ohm"] for point in grid]
    print(
        json.dumps(
            {
                "design_count": len(grid),
                "L_mH_range": [min(inductance_values), max(inductance_values)],
                "R_ohm_range": [min(resistance_values), max(resistance_values)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
