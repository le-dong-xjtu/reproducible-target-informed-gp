from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, NullFormatter
import numpy as np
from scipy.interpolate import PchipInterpolator

ESSAY = Path(__file__).resolve().parents[1]
PROJECT = ESSAY.parent
DATA = ESSAY / "data"
FIGURES = ESSAY / "figures" / "conference"
WIRE_WIDTH = 0.68
LIGHT_BLUE = "#9ECAE9"
DARK_BLUE = "#2171B5"
LIGHT_ORANGE = "#FDD0A2"
DARK_ORANGE = "#D94801"
OUTPUT_NAMES = {
    1: "fig01_surrogate_comparison.pdf",
    2: "fig02_gp_vs_sa.pdf",
    3: "fig03_acquisition_comparison.pdf",
    4: "fig04_initialization_reliability.pdf",
    5: "fig05_numerical_wpt_geometry.pdf",
    6: "fig06_maxwell_selected_geometry.pdf",
}


def read_csv(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_json(path):
    return json.loads(path.read_text())


def save_figure(fig, number, padding=0.02):
    path = FIGURES / OUTPUT_NAMES[number]
    fig.savefig(path, bbox_inches="tight", pad_inches=padding)
    print(path)


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render_figure_1() -> None:
    rows = benchmark_rows(
        "results/coil_2d_relative_tolerance/surrogate_family_comparison_budget27/results.csv"
    )
    labels = {
        "gp_random": "GP, Random Sampling",
        "gp_active": "GP, Uncertainty Sampling",
        "ensemble_random": "DE, Random Sampling",
        "ensemble_active": "DE, Uncertainty Sampling",
    }
    colors = {
        "gp_random": LIGHT_BLUE,
        "gp_active": DARK_BLUE,
        "ensemble_random": LIGHT_ORANGE,
        "ensemble_active": DARK_ORANGE,
    }
    styles = {
        "gp_random": "--",
        "gp_active": "-",
        "ensemble_random": "--",
        "ensemble_active": "-",
    }
    target_means = {}
    for method in labels:
        grouped = defaultdict(list)
        for row in rows:
            if row["method"] == method:
                grouped[int(float(row["target_mH"]))].append(
                    float(row["bounded_loss_u"])
                )
        target_means[method] = [
            (target, float(np.mean(values)))
            for (target, values) in sorted(grouped.items())
        ]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.55))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.1, top=0.97, hspace=0.42)
    for method, points in target_means.items():
        x, y = map(np.asarray, zip(*points))
        axes[0].plot(
            x,
            y,
            lw=1.2,
            marker="o",
            markersize=2.0,
            linestyle=styles[method],
            color=colors[method],
            label=labels[method],
        )
    axes[0].set(ylabel="Mean bounded loss, $u$", ylim=(-0.03, 1.26))
    axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    axes[0].set_xlabel("Target inductance (mH)", labelpad=2.5)
    axes[0].grid(color="#D9D9D9", lw=0.55)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.56, 0.95),
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        ncol=2,
        fontsize=6.2,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.28,
    )
    aggregate_labels = ("All-case mean", "Trimmed mean", "Worst-seed mean")
    positions = np.arange(len(aggregate_labels))
    width = 0.19
    offsets = np.arange(len(labels)) - (len(labels) - 1) / 2
    summaries = {
        method: loss_summary([row for row in rows if row["method"] == method])
        for method in labels
    }
    for offset, method in zip(offsets, labels):
        axes[1].bar(
            positions + offset * width,
            summaries[method],
            width=width,
            color=colors[method],
            label=labels[method],
        )
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.001, 1.0)
    axes[1].set_yticks([0.001, 0.01, 0.1, 1.0])
    axes[1].yaxis.set_major_formatter(LogFormatterMathtext())
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xticks(positions, aggregate_labels)
    axes[1].tick_params(axis="both", labelsize=8.2)
    axes[1].set_ylabel("Bounded loss, $u$ (log scale)")
    axes[1].grid(axis="y", color="#D9D9D9", lw=0.55, which="major")
    axes[1].grid(axis="y", color="#EDEDED", lw=0.35, which="minor")
    save_figure(fig, 1, padding=0.02)
    plt.close(fig)


def direct_rows_by_target() -> dict[str, list[dict[str, float]]]:
    rows = read_csv("direct_optimizer_summary_trimmed.csv")
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for r in rows:
        grouped[r["method"]].append(
            {
                "target": float(r["target_mH"]),
                "u": float(r["trimmed_bounded_loss_u"]),
                "cost": float(r["trimmed_true_evaluations"]),
            }
        )
    return grouped


def render_figure_2() -> None:
    rows = benchmark_rows(
        "results/coil_2d_relative_tolerance/surrogate_family_comparison_budget27/results.csv"
    )
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] == "gp_active":
            grouped[int(float(row["target_mH"]))].append(float(row["bounded_loss_u"]))
    gp = [
        {"target": target, "u": float(np.mean(sorted(values)[1:-1])), "cost": 27.0}
        for (target, values) in sorted(grouped.items())
    ]
    sa = direct_rows_by_target()["simulated_annealing"]
    sa_color = "#D95F02"
    gp_color = DARK_BLUE
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.45), sharex=True)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.97, hspace=0.18)
    for values, label, color in (
        (sa, "Simulated annealing (SA)", sa_color),
        (gp, "Reusable active GP ($B=27$)", gp_color),
    ):
        x = np.asarray([v["target"] for v in values], dtype=float)
        y = np.asarray([v["u"] for v in values], dtype=float)
        dense_x = np.linspace(x.min(), x.max(), 400)
        axes[0].plot(dense_x, PchipInterpolator(x, y)(dense_x), lw=1.2, color=color)
        axes[0].scatter(x, y, s=6, color=color, zorder=3)
        axes[0].plot(
            [], [], lw=1.2, marker="o", markersize=2.5, color=color, label=label
        )
    sa_x = np.asarray([v["target"] for v in sa], dtype=float)
    sa_cost = np.asarray([v["cost"] for v in sa], dtype=float)
    dense_sa_x = np.linspace(sa_x.min(), sa_x.max(), 400)
    axes[1].plot(
        dense_sa_x, PchipInterpolator(sa_x, sa_cost)(dense_sa_x), lw=1.2, color=sa_color
    )
    axes[1].scatter(sa_x, sa_cost, s=6, color=sa_color, zorder=3)
    axes[1].plot(
        [],
        [],
        lw=1.2,
        marker="o",
        markersize=2.5,
        color=sa_color,
        label="SA: independent search",
    )
    gp_targets = [v["target"] for v in gp]
    axes[1].plot(
        gp_targets,
        [27] * len(gp_targets),
        color=gp_color,
        lw=1.2,
        marker="o",
        markersize=2.5,
        label="Reusable active GP ($B=27$)",
        zorder=3,
    )
    axes[0].set(ylabel="Bounded loss, $u$", ylim=(-0.01, 0.17))
    axes[1].set(
        xlabel="Target inductance (mH)", ylabel="Model evaluations", ylim=(0, 165)
    )
    axes[0].tick_params(axis="both", labelsize=8.0)
    axes[0].grid(color="#D9D9D9", lw=0.55)
    axes[1].tick_params(axis="both", labelsize=8.0)
    axes[1].grid(color="#D9D9D9", lw=0.55)
    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.96),
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        ncol=1,
        fontsize=6.4,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.28,
    )
    save_figure(fig, 2, padding=0.02)
    plt.close(fig)


def benchmark_rows(relative_path: str) -> list[dict[str, str]]:
    with (PROJECT / relative_path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def loss_summary(rows: list[dict[str, str]]) -> tuple[float, float, float]:
    by_target: dict[int, list[float]] = defaultdict(list)
    by_seed: dict[int, list[float]] = defaultdict(list)
    all_losses = []
    for row in rows:
        loss = float(row["bounded_loss_u"])
        all_losses.append(loss)
        by_target[int(float(row["target_mH"]))].append(loss)
        by_seed[int(row["seed"])].append(loss)
    trimmed = [np.mean(sorted(values)[1:-1]) for values in by_target.values()]
    worst_seed = max((np.mean(values) for values in by_seed.values()))
    return (float(np.mean(all_losses)), float(np.mean(trimmed)), float(worst_seed))


def render_figure_3() -> None:
    methods = {
        "Reusable active GP": benchmark_rows(
            "results/coil_2d_relative_tolerance/global_dual_gp_budget27/alpha_0p5/results.csv"
        ),
        "GP-cEI": benchmark_rows(
            "results/coil_2d_relative_tolerance/cei_budget27/cei_results.csv"
        ),
        "GP-cLCB": benchmark_rows(
            "results/coil_2d_relative_tolerance/standard_bo_budget27/clcb_wl1_wg0_a0p5/results.csv"
        ),
        "GP-IVR": benchmark_rows(
            "results/coil_2d_relative_tolerance/standard_bo_budget27/ivr_wl1_wg0_a0p5/results.csv"
        ),
    }
    target_means = {}
    for name, rows in methods.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[int(float(row["target_mH"]))].append(float(row["bounded_loss_u"]))
        target_means[name] = [
            (target, float(np.mean(values)))
            for (target, values) in sorted(grouped.items())
        ]
    summary = {name: loss_summary(rows) for (name, rows) in methods.items()}
    colors = {
        "Reusable active GP": LIGHT_BLUE,
        "GP-cEI": DARK_BLUE,
        "GP-cLCB": LIGHT_ORANGE,
        "GP-IVR": DARK_ORANGE,
    }
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.55))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.1, top=0.97, hspace=0.42)
    for name, points in target_means.items():
        x, y = map(np.asarray, zip(*points))
        axes[0].plot(
            x, y, lw=1.2, marker="o", markersize=2.0, color=colors[name], label=name
        )
    axes[0].set(
        xlabel="Target inductance (mH)",
        ylabel="Mean bounded loss, $u$",
        ylim=(-0.003, 0.038),
    )
    axes[0].set_yticks([0.0, 0.01, 0.02, 0.03])
    axes[0].set_xlabel("Target inductance (mH)", labelpad=2.5)
    axes[0].tick_params(axis="both", labelsize=8.0)
    axes[0].grid(color="#D9D9D9", lw=0.55)
    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.96),
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        ncol=2,
        fontsize=6.2,
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.28,
    )
    labels = ("All-case mean", "Trimmed mean", "Worst-seed mean")
    positions = np.arange(len(labels))
    width = 0.155
    offsets = np.arange(len(methods)) - (len(methods) - 1) / 2
    for offset, name in zip(offsets, methods):
        axes[1].bar(
            positions + offset * width,
            summary[name],
            width=width,
            color=colors[name],
            label=name,
        )
    axes[1].set_yscale("log")
    axes[1].set_ylim(4e-05, 0.012)
    axes[1].set_yticks([0.0001, 0.001, 0.01])
    axes[1].yaxis.set_major_formatter(LogFormatterMathtext())
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xticks(positions, labels)
    axes[1].tick_params(axis="both", labelsize=8.0)
    axes[1].set_ylabel("Bounded loss, $u$ (log scale)")
    axes[1].grid(axis="y", color="#D9D9D9", lw=0.55, which="major")
    axes[1].grid(axis="y", color="#EDEDED", lw=0.35, which="minor")
    save_figure(fig, 3, padding=0.02)
    plt.close(fig)


def render_figure_4() -> None:
    folder = DATA / "wpt_starting_policy_screen_uncapped"
    results = [load_json(folder / policy / "result.json") for policy in
               ("deterministic", "random6_20261019")]
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.97)
    for result, color, label in zip(results, (DARK_BLUE, LIGHT_BLUE),
                                    ("Deterministic center", "Random-six (20261019)")):
        with (folder / result["policy"] / "trace.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        calls = np.array([int(row["evaluation"]) for row in rows])
        mutual = np.array([float(row["M_nH"]) for row in rows])
        n_model = result["n_model_building"]
        ax.plot(calls[:n_model], mutual[:n_model], color=color, lw=1.2,
                marker="o", ms=2.0, label=label)
        ax.plot(calls[n_model - 1:], mutual[n_model - 1:], color=color, lw=1.2)
        ax.scatter(calls[n_model:], mutual[n_model:], color=color,
                   marker="s", s=8, linewidths=0.4, zorder=4)
        assert result["stopped_naturally"]
    ax.axhline(1500, color=DARK_ORANGE, ls=(0, (2, 2)), lw=1.4,
               zorder=5, label="Target: 1500 nH")
    ax.grid(color="#D9D9D9", lw=0.55)
    ax.tick_params(axis="both", labelsize=8.0)
    ax.set_xlabel("Total evaluations", labelpad=2.5)
    ax.set_ylabel("Mutual inductance (nH)")
    ax.set_xlim(0, max(r["total_evaluations"] for r in results) + 2)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.96),
              frameon=True, facecolor="white", edgecolor="0.85",
              fontsize=6.2, handlelength=1.5, handletextpad=0.4, borderpad=0.28)
    save_figure(fig, 4)
    plt.close(fig)
    (FIGURES / "fig04_reliability_evidence.json").write_text(json.dumps({
        "source": str(folder), "results": results,
        "selection": "User-selected seed 20261019 from the ten-seed screen; not the worst-resistance seed.",
        "maximum_automatic_evaluations": None,
    }, indent=2) + "\n")


def _draw_polygon_coil(
    ax, ns: int, nt: int, spacing: float, inner_radius: float, z: float, color: str
) -> None:
    angles = 2 * np.pi * np.arange(ns) / ns
    unit_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    for turn in range(nt):
        radius = inner_radius + turn * (spacing + WIRE_WIDTH)
        vertices = radius * unit_vertices
        xy = np.vstack((vertices, vertices[0]))
        ax.plot(
            xy[:, 0], xy[:, 1], np.full(len(xy), z), color=color, lw=0.85, alpha=0.9
        )


def render_figure_5() -> None:
    result = load_json(DATA / "wpt_8d_automated_cei" / "result.json")
    ns_t, nt_t, s_t, ri_t, ns_r, nt_r, s_r, ri_r = result["design"]
    fig = plt.figure(figsize=(3.5, 2.75))
    ax = fig.add_subplot(111, projection="3d")
    _draw_polygon_coil(ax, round(ns_t), round(nt_t), s_t, ri_t, 0, "#d95f02")
    _draw_polygon_coil(ax, round(ns_r), round(nt_r), s_r, ri_r, 70, "#1b9e77")
    ax.plot([0, 0], [0, 0], [0, 70], color="0.35", ls="--", lw=1)
    ax.text(4, 4, 34, "$70$ mm", fontsize=7.2)
    ax.set_xlabel("$x$ (mm)", labelpad=1)
    ax.set_ylabel("$y$ (mm)", labelpad=1)
    ax.set_xticks([-60, 0, 60])
    ax.set_yticks([-60, 0, 60])
    ax.set_zticks([0, 70])
    ax.tick_params(axis="x", pad=-2, labelsize=7.2)
    ax.tick_params(axis="y", pad=-2, labelsize=7.2)
    ax.tick_params(axis="z", pad=1, labelsize=7.2)
    ax.text2D(
        1.06,
        0.53,
        "$z$ (mm)",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=8.5,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#d95f02", lw=1.4, label="Tx: 9 sides, 7 turns"),
            Line2D([0], [0], color="#1b9e77", lw=1.4, label="Rx: 7 sides, 7 turns"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
        fontsize=6.4,
        borderpad=0.3,
        handlelength=1.5,
    )
    ax.set_box_aspect((1, 1, 0.62))
    ax.view_init(elev=23, azim=-58)
    ax.grid(False)
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.08, top=0.96)
    save_figure(fig, 5, padding=0.06)
    plt.close(fig)


def render_figure_6():
    source = ESSAY / "figures" / "fig07_maxwell_selected_geometry.png"
    screenshot = plt.imread(source)
    height = 3.5 * screenshot.shape[0] / screenshot.shape[1]
    fig, ax = plt.subplots(figsize=(3.5, height))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(screenshot, interpolation="none")
    ax.set_axis_off()
    save_figure(fig, 6)
    plt.close(fig)


RENDERERS = {
    1: render_figure_1,
    2: render_figure_2,
    3: render_figure_3,
    4: render_figure_4,
    5: render_figure_5,
    6: render_figure_6,
}


def main():
    global FIGURES
    parser = argparse.ArgumentParser(
        description="Generate the six conference figure PDFs."
    )
    parser.add_argument(
        "--figures", type=int, choices=range(1, 7), nargs="+", default=list(range(1, 7))
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    FIGURES = args.output_dir.resolve()
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure()
    for number in dict.fromkeys(args.figures):
        RENDERERS[number]()


if __name__ == "__main__":
    main()
