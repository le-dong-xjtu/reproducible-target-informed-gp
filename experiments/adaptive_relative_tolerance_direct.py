from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_surrogate_targets import ROOT, truth_grid

EPSILON = .025
SEEDS = tuple(range(20261015, 20261025))
TARGETS = tuple(range(1, 88))
OUTPUT = ROOT / "results" / "coil_2d_relative_tolerance" / "direct_optimizer_baselines"
ALGORITHMS = ("simulated_annealing", "genetic_algorithm", "particle_swarm")


class Search:
    def __init__(self, grid, target, seed):
        self.grid, self.target, self.rng = grid, target, random.Random(seed)
        self.by_point = {(p["T"], p["n"]): p for p in grid}
        self.cache = {}

    def feasible(self, point):
        return abs(point["L_mH"] - self.target) / self.target <= EPSILON

    def violation(self, point):
        return max(0., abs(point["L_mH"] - self.target) / self.target - EPSILON)

    def key(self, point):
        return (0, point["R_ohm"]) if self.feasible(point) else (1, self.violation(point), point["R_ohm"])

    def evaluate(self, T, n):
        point = (min(10, max(1, round(T))), min(130, max(20, round(n))))
        if point not in self.cache:
            self.cache[point] = self.by_point[point]
        return self.cache[point]

    def random_unseen(self):
        remaining = [point for point in self.by_point if point not in self.cache]
        if not remaining:
            return min(self.cache.values(), key=self.key)
        return self.evaluate(*self.rng.choice(remaining))

    def unique_candidate(self, T, n):
        point = (min(10, max(1, round(T))), min(130, max(20, round(n))))
        return self.random_unseen() if point in self.cache else self.evaluate(*point)

    def best_feasible(self):
        valid = [point for point in self.cache.values() if self.feasible(point)]
        return min(valid, key=lambda point: point["R_ohm"]) if valid else None


def stalled(previous, current, count):
    if previous is None and current is not None:
        return current["R_ohm"], 0
    if current is None:
        return previous, count
    if current["R_ohm"] <= previous * .99:
        return current["R_ohm"], 0
    return previous, count + 1


def simulated_annealing(search):
    current = search.random_unseen()
    temperature, best_r, stale, cycles = 1.0, None, 0, 0
    while len(search.cache) < len(search.grid) and stale < 5:
        for _ in range(10):
            candidate = search.unique_candidate(
                current["T"] + search.rng.choice((-2, -1, 0, 1, 2)),
                current["n"] + search.rng.choice((-8, -4, -2, -1, 1, 2, 4, 8)),
            )
            merit = (0 if search.feasible(current) else 100 + 100 * search.violation(current)) + current["R_ohm"] / 100
            trial = (0 if search.feasible(candidate) else 100 + 100 * search.violation(candidate)) + candidate["R_ohm"] / 100
            if trial <= merit or search.rng.random() < math.exp((merit - trial) / max(temperature, 1e-9)):
                current = candidate
        cycles += 1
        best = search.best_feasible()
        best_r, stale = stalled(best_r, best, stale)
        temperature *= .90
    return search.best_feasible(), cycles, "five temperature stages without a ≥1% feasible-R improvement"


def genetic_algorithm(search):
    population = [search.random_unseen() for _ in range(20)]
    best_r, stale, cycles = None, 0, 0
    while len(search.cache) < len(search.grid) and stale < 5:
        population.sort(key=search.key)
        children = []
        for _ in range(20):
            a = min(search.rng.sample(population, 4), key=search.key)
            b = min(search.rng.sample(population, 4), key=search.key)
            T = a["T"] if search.rng.random() < .5 else b["T"]
            n = a["n"] if search.rng.random() < .5 else b["n"]
            if search.rng.random() < .45: T += search.rng.choice((-3, -2, -1, 1, 2, 3))
            if search.rng.random() < .60: n += search.rng.choice((-12, -8, -4, -2, -1, 1, 2, 4, 8, 12))
            children.append(search.unique_candidate(T, n))
        population = sorted(population + children, key=search.key)[:20]
        cycles += 1
        best_r, stale = stalled(best_r, search.best_feasible(), stale)
    return search.best_feasible(), cycles, "five generations without a ≥1% feasible-R improvement"


def particle_swarm(search):
    particles = []
    for _ in range(20):
        point = search.random_unseen()
        particles.append({"pos": [float(point["T"]), float(point["n"])],
                          "vel": [search.rng.uniform(-1, 1), search.rng.uniform(-8, 8)],
                          "best": point})
    global_best = min((particle["best"] for particle in particles), key=search.key)
    best_r, stale, cycles = None, 0, 0
    while len(search.cache) < len(search.grid) and stale < 5:
        for particle in particles:
            for dim in range(2):
                own = particle["best"]["T"] if dim == 0 else particle["best"]["n"]
                global_value = global_best["T"] if dim == 0 else global_best["n"]
                particle["vel"][dim] = (.729 * particle["vel"][dim] +
                    1.49445 * search.rng.random() * (own - particle["pos"][dim]) +
                    1.49445 * search.rng.random() * (global_value - particle["pos"][dim]))
                particle["pos"][dim] += particle["vel"][dim]
            point = search.unique_candidate(*particle["pos"])
            if search.key(point) < search.key(particle["best"]):
                particle["best"] = point
            if search.key(point) < search.key(global_best):
                global_best = point
        cycles += 1
        best_r, stale = stalled(best_r, search.best_feasible(), stale)
    return search.best_feasible(), cycles, "five swarm iterations without a ≥1% feasible-R improvement"


RUNNERS = {"simulated_annealing": simulated_annealing, "genetic_algorithm": genetic_algorithm,
           "particle_swarm": particle_swarm}


def append(path, rows, fields):
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new: writer.writeheader()
        writer.writerows(rows)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    grid = truth_grid(json.loads((ROOT / "config" / "coil_problem.json").read_text()))
    fields = ["method", "target_mH", "seed", "qualified", "relative_resistance_regret",
              "resistance_increase_percent", "bounded_loss_u", "true_evaluations", "cycles",
              "stop_reason", "decision_T", "decision_n", "decision_L_mH", "decision_R_ohm"]
    rows = []
    for target in TARGETS:
        optimum = min((point for point in grid if abs(point["L_mH"] - target) / target <= EPSILON),
                      key=lambda point: point["R_ohm"])
        target_rows = []
        for seed in SEEDS:
            for name in ALGORITHMS:
                search = Search(grid, target, seed)
                decision, cycles, reason = RUNNERS[name](search)
                if decision is None:
                    regret, loss = None, 1.
                else:
                    regret = decision["R_ohm"] / optimum["R_ohm"] - 1
                    loss = regret / (1 + regret)
                target_rows.append({
                    "method": name, "target_mH": target, "seed": seed, "qualified": decision is not None,
                    "relative_resistance_regret": regret,
                    "resistance_increase_percent": None if regret is None else 100 * regret,
                    "bounded_loss_u": loss, "true_evaluations": len(search.cache), "cycles": cycles,
                    "stop_reason": reason,
                    "decision_T": None if decision is None else decision["T"],
                    "decision_n": None if decision is None else decision["n"],
                    "decision_L_mH": None if decision is None else decision["L_mH"],
                    "decision_R_ohm": None if decision is None else decision["R_ohm"],
                })
        append(OUTPUT / "results.csv", target_rows, fields)
        rows.extend(target_rows)
        print(f"checkpoint: L={target} mH", flush=True)
    grouped = defaultdict(list)
    for row in rows: grouped[(row["method"], row["target_mH"])].append(row)
    summary = []
    for method in ALGORITHMS:
        for target in TARGETS:
            values = grouped[(method, target)]
            losses = sorted(float(value["bounded_loss_u"]) for value in values)
            costs = sorted(int(value["true_evaluations"]) for value in values)
            summary.append({"method": method, "target_mH": target,
                            "trimmed_bounded_loss_u": sum(losses[1:-1]) / 8,
                            "trimmed_true_evaluations": sum(costs[1:-1]) / 8,
                            "qualification_rate_percent": 10 * sum(value["qualified"] for value in values)})
    append(OUTPUT / "summary_by_target.csv", summary, list(summary[0]))
    labels = {"simulated_annealing": "Simulated annealing", "genetic_algorithm": "Genetic algorithm",
              "particle_swarm": "Particle swarm"}
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    figure.subplots_adjust(bottom=.09, hspace=.04)
    for method, label in labels.items():
        values = [row for row in summary if row["method"] == method]
        axes[0].plot(TARGETS, [row["trimmed_bounded_loss_u"] for row in values], label=label)
        axes[1].plot(TARGETS, [row["trimmed_true_evaluations"] for row in values], label=label)
    axes[0].set(ylabel="Trimmed bounded loss u (lower is better)", ylim=(-.02, 1.02),
                title="Adaptive direct optimizers, relative L tolerance ±2.5%")
    axes[1].set(xlabel="Target inductance (mH)", ylabel="Trimmed true evaluations")
    for axis in axes: axis.grid(alpha=.25)
    axes[0].legend(ncol=3)
    figure.text(.5, .015, "u = r/(1+r), r = R/R_exhaustive − 1; infeasible = 1. "
                           "Each trace drops the highest and lowest of 10 seeds.", ha="center", fontsize=8)
    figure.savefig(OUTPUT / "bounded_loss_and_cost_by_target.png", dpi=180)
    figure.savefig(OUTPUT / "bounded_loss_and_cost_by_target.svg")
    plt.close(figure)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "relative_inductance_tolerance": EPSILON,
        "stopping": "Feasible result required; then stop after five cycles without a ≥1% reduction in best feasible R. Safety cap: complete 1,110-point grid.",
        "loss": "u=r/(1+r), r=R/R_exhaustive-1; infeasible=1",
        "aggregation": "Discard highest and lowest of 10 seeds per target/method, average remaining eight.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
