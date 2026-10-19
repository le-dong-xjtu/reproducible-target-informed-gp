import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_surrogate_targets import ROOT, truth_grid

EPSILON = 0.025
SEEDS = tuple(range(20261015, 20261025))
TARGETS = tuple(range(1, 88))
OUTPUT = ROOT / "results" / "coil_2d_relative_tolerance" / "direct_optimizer_baselines"
ALGORITHMS = ("simulated_annealing", "genetic_algorithm", "particle_swarm")


class Search:
    def __init__(self, grid, target, seed):
        self.grid = grid
        self.target = target
        self.rng = random.Random(seed)
        self.by_point = {(point["T"], point["n"]): point for point in grid}
        self.cache = {}

    def feasible(self, point):
        return abs(point["L_mH"] - self.target) / self.target <= EPSILON

    def violation(self, point):
        return max(0.0, abs(point["L_mH"] - self.target) / self.target - EPSILON)

    def key(self, point):
        if self.feasible(point):
            return (0, point["R_ohm"])
        return (1, self.violation(point), point["R_ohm"])

    def evaluate(self, layers, turns):
        point = (min(10, max(1, round(layers))), min(130, max(20, round(turns))))
        if point not in self.cache:
            self.cache[point] = self.by_point[point]
        return self.cache[point]

    def random_unseen(self):
        remaining = [point for point in self.by_point if point not in self.cache]
        if not remaining:
            return min(self.cache.values(), key=self.key)
        return self.evaluate(*self.rng.choice(remaining))

    def unique_candidate(self, layers, turns):
        point = (min(10, max(1, round(layers))), min(130, max(20, round(turns))))
        if point in self.cache:
            return self.random_unseen()
        return self.evaluate(*point)

    def best_feasible(self):
        valid = [point for point in self.cache.values() if self.feasible(point)]
        return min(valid, key=lambda point: point["R_ohm"]) if valid else None


def stalled(previous, current, count):
    if previous is None and current is not None:
        return current["R_ohm"], 0
    if current is None:
        return previous, count
    if current["R_ohm"] <= previous * 0.99:
        return current["R_ohm"], 0
    return previous, count + 1


def simulated_annealing(search):
    current = search.random_unseen()
    temperature = 1.0
    best_r = None
    stale = 0
    cycles = 0
    while len(search.cache) < len(search.grid) and stale < 5:
        for _ in range(10):
            candidate = search.unique_candidate(
                current["T"] + search.rng.choice((-2, -1, 0, 1, 2)),
                current["n"] + search.rng.choice((-8, -4, -2, -1, 1, 2, 4, 8)),
            )
            merit = (
                0
                if search.feasible(current)
                else 100 + 100 * search.violation(current)
            ) + current["R_ohm"] / 100
            trial = (
                0
                if search.feasible(candidate)
                else 100 + 100 * search.violation(candidate)
            ) + candidate["R_ohm"] / 100
            if trial <= merit or search.rng.random() < math.exp(
                (merit - trial) / max(temperature, 1e-9)
            ):
                current = candidate
        cycles += 1
        best = search.best_feasible()
        best_r, stale = stalled(best_r, best, stale)
        temperature *= 0.90
    return (
        search.best_feasible(),
        cycles,
        "five temperature stages without a >=1% feasible-R improvement",
    )


def genetic_algorithm(search):
    population = [search.random_unseen() for _ in range(20)]
    best_r = None
    stale = 0
    cycles = 0
    while len(search.cache) < len(search.grid) and stale < 5:
        population.sort(key=search.key)
        children = []
        for _ in range(20):
            parent_a = min(search.rng.sample(population, 4), key=search.key)
            parent_b = min(search.rng.sample(population, 4), key=search.key)
            layers = (
                parent_a["T"] if search.rng.random() < 0.5 else parent_b["T"]
            )
            turns = (
                parent_a["n"] if search.rng.random() < 0.5 else parent_b["n"]
            )
            if search.rng.random() < 0.45:
                layers += search.rng.choice((-3, -2, -1, 1, 2, 3))
            if search.rng.random() < 0.60:
                turns += search.rng.choice((-12, -8, -4, -2, -1, 1, 2, 4, 8, 12))
            children.append(search.unique_candidate(layers, turns))
        population = sorted(population + children, key=search.key)[:20]
        cycles += 1
        best_r, stale = stalled(best_r, search.best_feasible(), stale)
    return (
        search.best_feasible(),
        cycles,
        "five generations without a >=1% feasible-R improvement",
    )


def particle_swarm(search):
    particles = []
    for _ in range(20):
        point = search.random_unseen()
        particles.append(
            {
                "pos": [float(point["T"]), float(point["n"])],
                "vel": [search.rng.uniform(-1, 1), search.rng.uniform(-8, 8)],
                "best": point,
            }
        )
    global_best = min((particle["best"] for particle in particles), key=search.key)
    best_r = None
    stale = 0
    cycles = 0
    while len(search.cache) < len(search.grid) and stale < 5:
        for particle in particles:
            for dim in range(2):
                own = particle["best"]["T"] if dim == 0 else particle["best"]["n"]
                global_value = (
                    global_best["T"] if dim == 0 else global_best["n"]
                )
                particle["vel"][dim] = (
                    0.729 * particle["vel"][dim]
                    + 1.49445 * search.rng.random() * (own - particle["pos"][dim])
                    + 1.49445
                    * search.rng.random()
                    * (global_value - particle["pos"][dim])
                )
                particle["pos"][dim] += particle["vel"][dim]
            point = search.unique_candidate(*particle["pos"])
            if search.key(point) < search.key(particle["best"]):
                particle["best"] = point
            if search.key(point) < search.key(global_best):
                global_best = point
        cycles += 1
        best_r, stale = stalled(best_r, search.best_feasible(), stale)
    return (
        search.best_feasible(),
        cycles,
        "five swarm iterations without a >=1% feasible-R improvement",
    )


RUNNERS = {
    "simulated_annealing": simulated_annealing,
    "genetic_algorithm": genetic_algorithm,
    "particle_swarm": particle_swarm,
}


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def append_rows(path, rows, fields):
    new_file = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT / "results.csv"
    grid = truth_grid(json.loads((ROOT / "config" / "coil_problem.json").read_text()))
    fields = [
        "method",
        "target_mH",
        "seed",
        "qualified",
        "relative_resistance_regret",
        "resistance_increase_percent",
        "bounded_loss_u",
        "true_evaluations",
        "cycles",
        "stop_reason",
        "decision_T",
        "decision_n",
        "decision_L_mH",
        "decision_R_ohm",
    ]
    existing = read_rows(results_path)
    completed = {
        (row["method"], int(float(row["target_mH"])), int(row["seed"]))
        for row in existing
    }
    for target in TARGETS:
        optimum = min(
            (
                point
                for point in grid
                if abs(point["L_mH"] - target) / target <= EPSILON
            ),
            key=lambda point: point["R_ohm"],
        )
        target_rows = []
        for seed in SEEDS:
            for name in ALGORITHMS:
                if (name, target, seed) in completed:
                    continue
                search = Search(grid, target, seed)
                decision, cycles, reason = RUNNERS[name](search)
                if decision is None:
                    regret, loss = None, 1.0
                else:
                    regret = decision["R_ohm"] / optimum["R_ohm"] - 1
                    loss = regret / (1 + regret)
                target_rows.append(
                    {
                        "method": name,
                        "target_mH": target,
                        "seed": seed,
                        "qualified": decision is not None,
                        "relative_resistance_regret": regret,
                        "resistance_increase_percent": (
                            None if regret is None else 100 * regret
                        ),
                        "bounded_loss_u": loss,
                        "true_evaluations": len(search.cache),
                        "cycles": cycles,
                        "stop_reason": reason,
                        "decision_T": None if decision is None else decision["T"],
                        "decision_n": None if decision is None else decision["n"],
                        "decision_L_mH": None if decision is None else decision["L_mH"],
                        "decision_R_ohm": (
                            None if decision is None else decision["R_ohm"]
                        ),
                    }
                )
        if target_rows:
            append_rows(results_path, target_rows, fields)
            print(f"checkpoint: L={target} mH", flush=True)
    rows = read_rows(results_path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(float(row["target_mH"])))].append(row)
    summary = []
    for method in ALGORITHMS:
        for target in TARGETS:
            values = grouped[(method, target)]
            losses = sorted(float(value["bounded_loss_u"]) for value in values)
            costs = sorted(int(value["true_evaluations"]) for value in values)
            qualified_count = sum(
                str(value["qualified"]).lower() == "true" for value in values
            )
            summary.append(
                {
                    "method": method,
                    "target_mH": target,
                    "trimmed_bounded_loss_u": sum(losses[1:-1]) / 8,
                    "trimmed_true_evaluations": sum(costs[1:-1]) / 8,
                    "qualification_rate_percent": 10 * qualified_count,
                }
            )
    summary_path = OUTPUT / "summary_by_target.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            {
                "relative_inductance_tolerance": EPSILON,
                "stopping": "Feasible result required; then stop after five cycles without a >=1% reduction in best feasible R. Safety cap: complete 1,110-point grid.",
                "loss": "u=r/(1+r), r=R/R_exhaustive-1; infeasible=1",
                "aggregation": "Discard highest and lowest of 10 seeds per target/method, average remaining eight.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"cases": len(rows), "summary_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
