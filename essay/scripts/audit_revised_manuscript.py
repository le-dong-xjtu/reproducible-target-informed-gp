#!/usr/bin/env python3
"""Reproduce integrity checks for the revised manuscript evidence package."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
ESSAY = ROOT / "essay"


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return value.strip().lower() == "true"


def assert_close(a, b, tolerance=2e-13):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance), (a, b)


def audit_u_from_regret():
    paths = [
        ROOT/"results/coil_2d_relative_tolerance/surrogate_family_comparison_budget27/results.csv",
        ESSAY/"data/direct_optimizer_results_raw_10seeds.csv",
        ROOT/"results/coil_2d_relative_tolerance/global_dual_gp_budget27/alpha_0p5/results.csv",
        ROOT/"results/coil_2d_relative_tolerance/cei_budget27/cei_results.csv",
        ROOT/"results/coil_2d_relative_tolerance/standard_bo_budget27/clcb_wl1_wg0_a0p5/results.csv",
        ROOT/"results/coil_2d_relative_tolerance/standard_bo_budget27/ivr_wl1_wg0_a0p5/results.csv",
    ]
    count = 0
    for path in paths:
        for row in rows(path):
            expected = (float(row["relative_resistance_regret"])
                        / (1+float(row["relative_resistance_regret"]))) if truth(row["qualified"]) else 1.
            assert_close(row["bounded_loss_u"], expected, 2e-14)
            count += 1
    return count


def coil_oracle():
    cfg = json.loads((ROOT/"config/coil_problem.json").read_text())
    wire, geometry = cfg["wire"], cfg["geometry"]
    pitch = wire["diameter_mm"]*wire["winding_factor"]
    values = []
    for layers in range(geometry["layers"]["min"], geometry["layers"]["max"]+1):
        for turns in range(geometry["turns_per_layer"]["min"], geometry["turns_per_layer"]["max"]+1):
            total = layers*turns
            height, thickness = turns*pitch, layers*pitch
            diameter = geometry["inner_diameter_mm"]+thickness
            inductance = .08*total**2*diameter**2*1e-4/(3*diameter+9*height+10*thickness)
            resistance = 4*wire["resistivity_ohm_mm"]*total*diameter/wire["diameter_mm"]**2
            values.append((inductance, resistance))
    return values


def audit_mixed_u_and_pairing():
    values = coil_oracle()
    paths = [
        ROOT/"results/coil_2d_relative_tolerance/mixed_uncertainty/development/results.csv",
        ROOT/"results/coil_2d_relative_tolerance/mixed_uncertainty/confirmation_b24_b30/results.csv",
    ]
    count = 0
    confirmation = None
    for path in paths:
        data = rows(path)
        for row in data:
            target = float(row["target_mH"])
            feasible = [r for l, r in values if abs(l-target) <= .025*target]
            expected = 1.
            if truth(row["qualified"]):
                regret = float(row["decision_R_ohm"])/min(feasible)-1
                expected = regret/(1+regret)
            assert_close(row["bounded_loss_u"], expected)
            assert int(row["budget"]) == int(row["n_initial"])+int(row["n_sequential"])+int(row["n_confirmation"])
            count += 1
        if "confirmation_b24_b30" in str(path):
            confirmation = data
    paired = defaultdict(dict)
    for row in confirmation:
        key = row["seed"], row["target_mH"], row["budget"]
        paired[key][row["policy"]] = row["selected_indices"].split(",")[:6]
    assert all(set(group) == {"cei", "cei_all"} and group["cei"] == group["cei_all"]
               for group in paired.values())
    return count, len(paired)


def audit_wpt_summary(folder):
    data, summary = rows(folder/"results.csv"), json.loads((folder/"summary.json").read_text())
    lookup = {item["policy"]: item for item in summary}
    grouped = defaultdict(list)
    for row in data:
        assert int(row["budget"]) == int(row["n_initial"])+int(row["n_sequential"])+int(row["n_confirmation"]) == 50
        grouped[row["policy"]].append(row)
    for policy, group in grouped.items():
        feasible = [row for row in group if truth(row["qualified"])]
        item = lookup[policy]
        assert_close(item["qualification_rate"], len(feasible)/len(group))
        resistance = np.asarray([float(row["best_R_ohm"]) for row in feasible])
        assert_close(item["mean_R_qualified_ohm"], resistance.mean())
        assert_close(item["std_R_qualified_ohm"], resistance.std(ddof=0))
        assert_close(item["mean_target_error_pct"],
                     np.mean([float(row["relative_target_error_pct"]) for row in group]))
    return len(data)


def audit_resistance_constrained_summary(folder):
    data, summary = rows(folder/"results.csv"), json.loads((folder/"summary.json").read_text())
    lookup = {item["policy"]: item for item in summary}
    grouped = defaultdict(list)
    for row in data:
        assert int(row["budget"]) == int(row["n_initial"])+int(row["n_sequential"])+int(row["n_confirmation"]) == 50
        assert truth(row["qualified"]) == (float(row["best_R_ohm"]) <= float(row["resistance_cap_ohm"])+1e-12)
        grouped[row["policy"]].append(row)
    for policy, group in grouped.items():
        feasible = [row for row in group if truth(row["qualified"])]
        item = lookup[policy]
        coupling = np.asarray([float(row["best_M_nH"]) for row in feasible])
        resistance = np.asarray([float(row["best_R_ohm"]) for row in feasible])
        assert_close(item["qualification_rate"], len(feasible)/len(group))
        assert_close(item["mean_best_M_nH"], coupling.mean())
        assert_close(item["std_best_M_nH"], coupling.std(ddof=0))
        assert_close(item["mean_R_ohm"], resistance.mean())
    return len(data)


def audit_wpt_budget_sweep(folder):
    data, summary = rows(folder/"results.csv"), json.loads((folder/"summary.json").read_text())
    assert len(data) == 2*10*51 and len(summary) == 2*51
    grouped = defaultdict(list)
    for row in data:
        assert int(row["budget"]) == int(row["n_initial"])+int(row["n_sequential"])+int(row["n_confirmation"])
        grouped[(row["policy"], int(row["budget"]))].append(row)
    for item in summary:
        group = grouped[(item["policy"], int(item["budget"]))]
        feasible = [row for row in group if truth(row["qualified"])]
        resistance = np.asarray([float(row["best_R_ohm"]) for row in feasible])
        assert len(group) == int(item["cases"]) == 10
        assert_close(item["qualification_rate"], len(feasible)/len(group))
        assert_close(item["mean_R_qualified_ohm"], resistance.mean())
        assert_close(item["std_R_qualified_ohm"], resistance.std(ddof=0))
    for policy in ("cei", "cei_all05"):
        for seed in sorted({int(row["seed"]) for row in data}):
            sequence = sorted((row for row in data
                               if row["policy"] == policy and int(row["seed"]) == seed),
                              key=lambda row: int(row["budget"]))
            histories = [json.loads(row["active_designs"]) for row in sequence]
            assert all(histories[i+1][:-1] == histories[i]
                       for i in range(len(histories)-1))
    for seed in sorted({int(row["seed"]) for row in data}):
        c = next(row for row in data if row["policy"] == "cei" and int(row["seed"]) == seed)
        m = next(row for row in data if row["policy"] == "cei_all05" and int(row["seed"]) == seed)
        assert json.loads(c["active_designs"])[:6] == json.loads(m["active_designs"])[:6]
    references = {
        "cei": ESSAY/"data/wpt_8d_mixed/results.csv",
        "cei_all05": ESSAY/"data/wpt_8d_low_auxiliary/results.csv",
    }
    for policy, path in references.items():
        original = {int(row["seed"]): row for row in rows(path) if row["policy"] == policy}
        replay = {int(row["seed"]): row for row in data
                  if row["policy"] == policy and int(row["budget"]) == 50}
        assert set(original) == set(replay)
        for seed in original:
            assert original[seed]["qualified"] == replay[seed]["qualified"]
            assert original[seed]["design"] == replay[seed]["design"]
            assert_close(original[seed]["best_M_nH"], replay[seed]["best_M_nH"])
            assert_close(original[seed]["best_R_ohm"], replay[seed]["best_R_ohm"])
    return len(data)


def audit_center_start_evidence():
    oracle = coil_oracle()
    baseline = {(int(row["seed"]), int(float(row["target_mH"]))): row
                for row in rows(ROOT/"results/coil_2d_relative_tolerance/cei_budget27/cei_results.csv")}
    folders = {
        "screen": ESSAY/"data/center_feasibility_smoke/results.csv",
        "low": ESSAY/"data/center_feasibility_low_targets/results.csv",
    }
    data = {name: rows(path) for name, path in folders.items()}
    assert {name: len(group) for name, group in data.items()} == {
        "screen": 300, "low": 40}
    for group in data.values():
        for row in group:
            assert int(row["budget"]) == (int(row["n_initial"])
                                           + int(row["n_sequential"])
                                           + int(row["n_confirmation"])) == 27
            target = int(row["target_mH"])
            feasible_r = [r for l, r in oracle if abs(l-target)/target <= .025]
            expected = 1.
            if truth(row["qualified"]):
                regret = float(row["decision_R_ohm"])/min(feasible_r)-1
                expected = regret/(1+regret)
            assert_close(row["bounded_loss_u"], expected)
    screen = defaultdict(list)
    for row in data["screen"]:
        screen[float(row["pre_uncertainty_weight"])].append(row)
    assert set(screen) == {.25, .5, .75}
    assert [sum(truth(row["exact"]) for row in screen[w]) for w in (.25, .5, .75)] == [100, 100, 99]
    assert all(truth(row["qualified"]) for row in data["low"])
    assert all(truth(row["exact"]) for row in data["low"])
    low_keys = [(int(row["seed"]), int(row["target_mH"])) for row in data["low"]]
    assert sum(float(baseline[key]["bounded_loss_u"]) <= 1e-12 for key in low_keys) == 37
    return sum(map(len, data.values()))


def audit_automatic_stopping():
    folders = {
        "stochastic_0p1": ESSAY/"data/center_cosine_stopping_smoke",
        "stochastic_1": ESSAY/"data/center_stopping_1pct_smoke",
        "deterministic_1": ESSAY/"data/center_stopping_1pct_deterministic_gp_k5",
    }
    expected_means = {
        "stochastic_0p1": 23.742857142857144,
        "stochastic_1": 22.59285714285714,
        "deterministic_1": 22.571428571428573,
    }
    count = 0
    for name, folder in folders.items():
        data = [row for row in rows(folder/"results.csv")
                if row["policy"] == "ordinary_stop"]
        assert len(data) == 140
        assert all(truth(row["qualified"]) and truth(row["exact"]) for row in data)
        assert all(int(row["n_confirmation"]) == 10 for row in data)
        calls = [int(row["oracle_calls"]) for row in data]
        assert_close(np.mean(calls), expected_means[name])
        summary = json.loads((folder/"summary.json").read_text())
        if name == "deterministic_1":
            assert summary["minimum_model_building"] == 5
        report = next(item for item in summary["results"]
                      if item["policy"] == "ordinary_stop")
        assert_close(report["mean_oracle_calls"], np.mean(calls))
        assert report["exact_optimum_cases"] == 140
        count += len(data)
    deterministic = rows(folders["deterministic_1"]/"results.csv")
    for target in {row["target_mH"] for row in deterministic}:
        group = [row for row in deterministic if row["target_mH"] == target]
        for field in ("model_building_ids", "confirmation_ids", "decision_T",
                      "decision_n", "oracle_calls"):
            assert len({row[field] for row in group}) == 1
    return count


def audit_automated_wpt():
    automatic = json.loads((ESSAY/"data/wpt_8d_automated_cei/result.json").read_text())
    control = json.loads((ESSAY/"data/wpt_8d_automated_cei_fixed_b50_control/result.json").read_text())
    assert automatic["qualified"] and automatic["stopped_naturally"]
    assert automatic["n_model_building"] == 34
    assert automatic["n_confirmation"] == 10
    assert automatic["total_evaluations"] == 44
    assert control["n_model_building"] == 40
    assert control["n_confirmation"] == 10
    assert control["total_evaluations"] == 50
    assert automatic["design"] == control["design"]
    for field in ("best_M_nH", "best_R_ohm", "relative_target_error_pct"):
        assert_close(automatic[field], control[field], tolerance=1e-12)
    manifest = json.loads((ESSAY/"data/wpt_8d_automated_cei/manifest.json").read_text())
    assert manifest["stop"]["minimum_model_building"] == 17
    assert abs(automatic["best_M_nH"]-1500)/1500 <= .025
    assert_close(automatic["best_R_ohm"], .3057735584037057)
    trace = rows(ESSAY/"data/wpt_8d_automated_cei/trace.csv")
    assert len(trace) == automatic["n_model_building"]
    assert next(int(row["model_call"]) for row in trace
                if abs(float(row["M_nH"])-1500)/1500 <= .025) == 12
    replay = ESSAY/"data/wpt_8d_automated_cei/first_unguarded_run"
    assert ((replay/"model_building_designs.json").read_bytes()
            == (ESSAY/"data/wpt_8d_automated_cei/model_building_designs.json").read_bytes())
    assert ((replay/"confirmation_designs.json").read_bytes()
            == (ESSAY/"data/wpt_8d_automated_cei/confirmation_designs.json").read_bytes())
    return len(trace)+automatic["n_confirmation"]+control["total_evaluations"]


def audit_heldout_stopping():
    folder = ESSAY/"data/center_stopping_heldout_73"
    data = rows(folder/"results.csv")
    summary = json.loads((folder/"summary.json").read_text())
    development = set(summary["split"]["development_targets_mH"])
    targets = {int(row["target_mH"]) for row in data}
    assert len(data) == len(targets) == 73
    assert not targets & development
    assert targets | development == set(range(1, 88))
    assert all(truth(row["qualified"]) and truth(row["exact"])
               and truth(row["stopped_naturally"]) for row in data)
    calls = np.asarray([int(row["oracle_calls"]) for row in data])
    report = summary["results"]
    assert summary["frozen_configuration"]["minimum_model_building"] == 5
    assert report["qualified_targets"] == report["exact_optimum_targets"] == 73
    assert_close(report["mean_total_calls"], calls.mean())
    assert_close(report["median_total_calls"], np.median(calls))
    assert report["minimum_total_calls"] == int(calls.min()) == 19
    assert report["maximum_total_calls"] == int(calls.max()) == 25
    return len(data)


def audit_multitarget_wpt():
    folder = ESSAY/"data/wpt_8d_multitarget_validation"
    data = rows(folder/"results.csv")
    manifest = json.loads((folder/"manifest.json").read_text())
    summary = {item["policy"]: item for item in
               json.loads((folder/"summary.json").read_text())}
    policies = {row["policy"] for row in data}
    targets = {float(row["target_M_nH"]) for row in data}
    assert policies == {"automated", "center_pf_cei_matched",
                        "conventional_cei_matched", "sobol_matched",
                        "forced_b50", "conventional_cei_b50"}
    assert targets == {float(value) for value in range(1000, 2001, 50)}
    assert manifest["targets_nH"] == [float(value) for value in range(1000, 2001, 50)]
    assert manifest["shared_gp"]["kernel"] == "Constant * isotropic Matern-5/2 + White"
    assert manifest["shared_gp"]["length_scale_bounds"] == [.03, 3.0]
    assert manifest["shared_gp"]["starts"] == 5
    assert len(data) == len(policies)*len(targets) == 126
    grouped = defaultdict(list)
    by_target = defaultdict(dict)
    for row in data:
        policy, target = row["policy"], float(row["target_M_nH"])
        grouped[policy].append(row); by_target[target][policy] = row
        assert int(row["total_evaluations"]) == (int(row["n_model_building"])
                                                  + int(row["n_confirmation"]))
        margin = .025*target-abs(float(row["best_M_nH"])-target)
        assert_close(row["feasibility_margin_nH"], margin)
        assert truth(row["qualified"]) == (margin >= -1e-12)
    for target, group in by_target.items():
        realized = int(group["automated"]["total_evaluations"])
        assert int(group["conventional_cei_matched"]["total_evaluations"]) == realized
        assert int(group["center_pf_cei_matched"]["total_evaluations"]) == realized
        assert int(group["sobol_matched"]["total_evaluations"]) == realized
        assert int(group["forced_b50"]["total_evaluations"]) == 50
        assert int(group["conventional_cei_b50"]["total_evaluations"]) == 50
    expected_qualified = {"automated": 21, "center_pf_cei_matched": 21,
                          "conventional_cei_matched": 21,
                          "sobol_matched": 10, "forced_b50": 21,
                          "conventional_cei_b50": 21}
    for policy, group in grouped.items():
        qualified = [row for row in group if truth(row["qualified"])]
        assert len(qualified) == expected_qualified[policy]
        item = summary[policy]
        assert item["qualified_cases"] == len(qualified)
        assert_close(item["mean_total_evaluations"],
                     np.mean([int(row["total_evaluations"]) for row in group]))
        assert_close(item["mean_R_qualified_ohm"],
                     np.mean([float(row["best_R_ohm"]) for row in qualified]))
    automatic = sorted(grouped["automated"], key=lambda row: float(row["target_M_nH"]))
    conventional = sorted(grouped["conventional_cei_matched"],
                          key=lambda row: float(row["target_M_nH"]))
    differences = np.asarray([float(a["best_R_ohm"])-float(c["best_R_ohm"])
                              for a, c in zip(automatic, conventional)])
    statistic, pvalue = wilcoxon(differences, zero_method="wilcox",
                                 alternative="two-sided", method="auto")
    assert_close(statistic, 83.0)
    assert_close(pvalue, .9133009306692093)
    assert sum(int(row["n_model_building"]) == 17 for row in automatic) == 4
    return len(data)


def audit_confirmation_ablation():
    folder = ESSAY/"data/confirmation_count_ablation"
    data = rows(folder/"results.csv")
    assert [int(row["confirmations"]) for row in data] == [0, 1, 2, 5, 10]
    assert all(int(row["cases"]) == 140 for row in data)
    assert all(int(row["qualified_cases"]) == 140 for row in data)
    assert all(int(row["exact_optimum_cases"]) == 140 for row in data)
    return len(data)


def audit_initialization_screen():
    folder = ESSAY/"data/initialization_screen_1000"
    data = rows(folder/"seed_initialization_1000x87.csv")
    points = rows(folder/"seed_initialization_points.csv")
    assert len(data) == 87000
    assert len(points) == 1000
    seeds = list(range(20261025, 20262025))
    assert sorted({int(row["seed"]) for row in data}) == seeds
    assert sorted({int(row["target_mH"]) for row in data}) == list(range(1, 88))
    assert sorted(int(row["seed"]) for row in points) == seeds

    oracle = coil_oracle()
    lookup = {}
    index = 0
    for layers in range(1, 11):
        for turns in range(20, 131):
            lookup[(layers, turns)] = oracle[index][0]
            index += 1
    expected = {}
    for row in points:
        inductances = [lookup[tuple(map(int, item.split(":")))]
                       for item in row["six_designs_T:n"].split(",")]
        for target in range(1, 88):
            errors = [abs(value-target)/target for value in inductances]
            expected[(int(row["seed"]), target)] = (min(errors), min(errors) <= .025)
    for row in data:
        key = int(row["seed"]), int(row["target_mH"])
        error, feasible = expected[key]
        assert_close(row["min_relative_L_error"], error, 1e-12)
        assert int(row["initial_feasible"]) == int(feasible)
    counts = defaultdict(int)
    for row in data:
        counts[int(row["target_mH"])] += int(row["initial_feasible"])
    assert {target: counts[target] for target in (2, 3, 4, 5, 87)} == {
        2: 34, 3: 65, 4: 32, 5: 72, 87: 22}
    return len(data)


def audit_worst_initialization_case():
    report = json.loads((ESSAY/"data/initialization_screen_1000"/
                         "worst_seed_end_to_end/result.json").read_text())
    assert report["recorded_seed"] == 20261105
    assert report["recorded_initial_points"] == [
        [10, 50], [10, 84], [8, 86], [8, 122], [6, 115], [10, 64]]
    baseline = report["random6_baseline"]
    automatic = report["deterministic_automatic"]
    assert baseline["qualified"] and baseline["exact"]
    assert automatic["qualified"] and automatic["exact"]
    assert baseline["first_feasible_evaluation"] == 12
    assert automatic["first_feasible_evaluation"] == 9
    assert baseline["budget"] == 27
    assert automatic["oracle_calls"] == 23
    assert baseline["decision"]["point"] == [5, 25]
    assert [automatic["decision_T"], automatic["decision_n"]] == [5, 25]
    return 1


def audit_random6_adaptive_stopping():
    folder = ESSAY/"data/random6_adaptive_stopping"
    data = rows(folder/"results.csv")
    summary = json.loads((folder/"summary.json").read_text())
    manifest = json.loads((folder/"manifest.json").read_text())
    assert len(data) == 870
    assert manifest["status"] == "complete"
    assert manifest["gp_optimizer"] == "deterministic Halton multistart K=5"
    assert manifest["budget"] == {
        "maximum_model_building": 17, "confirmation": 10, "maximum_total": 27}
    assert all(int(row["n_confirmation"]) == 10 for row in data)
    assert all(int(row["oracle_calls"]) == int(row["n_model_building"])+10
               and int(row["oracle_calls"]) <= 27 for row in data)
    assert sum(truth(row["qualified"]) for row in data) == 870
    assert sum(truth(row["exact"]) for row in data) == 856
    by_target = defaultdict(list)
    by_seed = defaultdict(list)
    for row in data:
        by_target[int(row["target_mH"])].append(row)
        by_seed[int(row["seed"])].append(row)
    assert set(by_target) == set(range(1, 88))
    assert all(len(group) == 10 for group in by_target.values())
    assert all(len(group) == 87 for group in by_seed.values())
    for group in by_seed.values():
        initial_sets = {tuple(json.loads(row["model_building_ids"])[:6]) for row in group}
        assert len(initial_sets) == 1
    report = summary["random6"]
    assert report["exact_cases"] == 856
    assert report["targets_with_seed_variable_decision"] == 8
    assert report["nonexact_targets_mH"] == [1, 2, 3, 4, 9, 11, 13, 22]
    paired = summary["paired_random_cases_vs_target_matched_deterministic"]
    assert [paired["deterministic_wins"], paired["ties"],
            paired["deterministic_losses"]] == [14, 856, 0]
    assert_close(report["mean_calls"], 22.570114942528736)
    assert_close(summary["deterministic_center"]["mean_calls"], 21.93103448275862)
    return len(data)


def audit_clcb_replay():
    folder = ESSAY/"data/clcb_correct_sign_replay"
    report = json.loads((folder/"verification.json").read_text())
    data = rows(folder/"results.csv")
    source = ROOT/"methods/target_informed_gp/gp_conventional_acquisition_baselines.py"
    assert report["status"] == "complete"
    assert report["mismatch_count"] == 0
    assert len(data) == report["fresh_rows"] == report["reference_rows"] == 870
    assert report["implementation"]["lower_bound"] == "mean_R - kappa*std_R"
    assert report["implementation"]["kappa"] == 2.0
    assert report["implementation"]["source_sha256"] == hashlib.sha256(
        source.read_bytes()).hexdigest()
    summary = report["fresh_replay_summary"]
    assert summary["exact_optimum_cases"] == 829
    assert_close(summary["all_case_mean_u"], .0027380531429969063)
    return len(data)


def audit_stopping_calibration():
    folder = ESSAY/"data/stopping_gp_calibration_heldout_73"
    data = rows(folder/"results.csv")
    summary = json.loads((folder/"summary.json").read_text())
    assert len(data) == summary["cases"] == 73
    assert all(truth(row["incumbent_exact_at_stop"]) for row in data)
    assert all(float(row["actual_available_relative_improvement"]) == 0 for row in data)
    feasible = summary["true_feasible_region_macro_mean"]
    expected = {
        "feasible_L_coverage_1sigma": .8078440965427267,
        "feasible_L_coverage_2sigma": .9646543715036867,
        "feasible_R_coverage_1sigma": .8211038961038961,
        "feasible_R_coverage_2sigma": .9625985886259858,
    }
    for key, value in expected.items():
        assert_close(feasible[key], value)
    quality = summary["stopping_quality"]
    assert quality["incumbent_exact_at_stop"] == 73
    assert quality["targets_with_actual_improvement_above_1pct"] == 0
    return len(data)


def audit_assets():
    required = [
        ESSAY/"coil_surrogate_ieee_draft.tex", ESSAY/"coil_surrogate_ieee_draft.pdf",
        ESSAY/"sources/elp2.70015.pdf",
        ESSAY/"scripts/render_figures.py", ESSAY/"scripts/render_mixed_results.py",
        ESSAY/"scripts/run_wpt_8d_mixed_benchmark.py",
        ESSAY/"scripts/run_wpt_8d_resistance_constrained.py",
        ESSAY/"scripts/run_wpt_8d_budget_sweep.py",
        ESSAY/"scripts/run_center_start_control.py",
        ESSAY/"scripts/run_center_feasibility_smoke.py",
        ESSAY/"scripts/run_center_feasibility_low_targets.py",
        ESSAY/"scripts/run_center_cosine_stopping_smoke.py",
        ESSAY/"scripts/run_automatic_gp_wpt8d.py",
        ESSAY/"scripts/run_wpt_8d_multitarget_validation.py",
        ESSAY/"scripts/run_automatic_gp_2d.py",
        ESSAY/"scripts/analyze_stopping_gp_calibration.py",
        ESSAY/"scripts/analyze_confirmation_ablation.py",
        ESSAY/"scripts/seed_initialization_screen.py",
        ESSAY/"scripts/verify_initialization_screen.py",
        ESSAY/"scripts/run_worst_initialization_case.py",
        ESSAY/"scripts/run_random_start_adaptive_stopping.py",
        ESSAY/"scripts/verify_clcb_replay.py",
        ESSAY/"scripts/render_automated_stopping.py",
        ESSAY/"scripts/render_wpt_selected_geometry.py",
        ESSAY/"data/center_feasibility_smoke/results.csv",
        ESSAY/"data/center_feasibility_low_targets/results.csv",
        ESSAY/"scripts/render_wpt_budget_sweep.py",
        ESSAY/"figures/fig07_wpt_budget_sweep_b30_b80.pdf",
        ESSAY/"figures/fig05_automated_stopping.pdf",
        ESSAY/"figures/fig06_wpt_selected_geometry.pdf",
        ESSAY/"data/center_stopping_heldout_73/results.csv",
        ESSAY/"data/wpt_8d_multitarget_validation/results.csv",
        ESSAY/"data/stopping_gp_calibration_heldout_73/results.csv",
        ESSAY/"data/confirmation_count_ablation/results.csv",
        ESSAY/"data/initialization_screen_1000/seed_initialization_1000x87.csv",
        ESSAY/"data/initialization_screen_1000/seed_initialization_points.csv",
        ESSAY/"data/initialization_screen_1000/worst_seed_end_to_end/result.json",
        ESSAY/"data/random6_adaptive_stopping/results.csv",
        ESSAY/"data/clcb_correct_sign_replay/verification.json",
    ]
    required += [ESSAY/"figures"/f"fig0{i}_{name}.pdf" for i, name in [
        (1, "global_surrogate_2x2"), (2, "direct_optimizers"),
        (3, "gaussian_process_vs_sa"), (4, "gp_acquisition_comparison"),
        (5, "mixed_uncertainty_budget"), (6, "wpt_8d_mixed")]]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, missing
    return len(required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numerical-only", action="store_true",
        help="run all retained numerical checks but skip the local manuscript/source asset inventory",
    )
    args = parser.parse_args()
    regret_rows = audit_u_from_regret()
    mixed_rows, pairs = audit_mixed_u_and_pairing()
    wpt_rows = sum(audit_wpt_summary(folder) for folder in [
        ESSAY/"data/wpt_8d_mixed", ESSAY/"data/wpt_8d_low_auxiliary",
        ESSAY/"data/wpt_8d_frozen_weight_sensitivity"])
    dual_rows = audit_resistance_constrained_summary(ESSAY/"data/wpt_8d_resistance_constrained")
    sweep_rows = audit_wpt_budget_sweep(ESSAY/"data/wpt_8d_budget_sweep_b30_b80")
    center_rows = audit_center_start_evidence()
    stopping_rows = audit_automatic_stopping()
    heldout_rows = audit_heldout_stopping()
    calibration_rows = audit_stopping_calibration()
    clcb_rows = audit_clcb_replay()
    automated_wpt_calls = audit_automated_wpt()
    multitarget_rows = audit_multitarget_wpt()
    confirmation_rows = audit_confirmation_ablation()
    initialization_rows = audit_initialization_screen()
    worst_case_rows = audit_worst_initialization_case()
    random_stopping_rows = audit_random6_adaptive_stopping()
    assets = None if args.numerical_only else audit_assets()
    print(f"PASS: u verified in {regret_rows+mixed_rows} rows; {pairs} paired initializations match")
    print(f"PASS: {wpt_rows} WPT rows reproduce stored summaries and B=50 accounting")
    print(f"PASS: {dual_rows} resistance-capped rows reproduce stored summaries and B=50 accounting")
    print(f"PASS: {sweep_rows} budget-sweep rows reproduce summaries, nested histories, and B=50 replay")
    print(f"PASS: {center_rows} deterministic-start rows reproduce budgets, losses, and headline counts")
    print(f"PASS: {stopping_rows} stopping rows reproduce exactness, budgets, means, and deterministic invariance")
    print(f"PASS: {heldout_rows} held-out targets reproduce frozen stopping claims")
    print(f"PASS: {calibration_rows} stopping-point GPs reproduce local calibration claims")
    print(f"PASS: {clcb_rows} fresh cLCB rows exactly reproduce Fig. 4 with the corrected sign")
    print(f"PASS: {automated_wpt_calls} automated/control WPT calls reproduce qualification, stopping, and replay identity")
    print(f"PASS: {multitarget_rows} multi-target WPT rows reproduce matched budgets and qualification")
    print(f"PASS: {confirmation_rows} confirmation-count ablations preserve development outcomes")
    print(f"PASS: {initialization_rows} initialization-only rows reproduce archived six-point coverage")
    print(f"PASS: {worst_case_rows} preselected worst-start trajectory reproduces the exact-optimum comparison")
    print(f"PASS: {random_stopping_rows} matched random-six adaptive-stopping rows reproduce stability claims")
    if assets is not None:
        print(f"PASS: {assets} required manuscript/source/code/figure assets are present")
    else:
        print("PASS: numerical audit complete; local manuscript/source asset inventory skipped")


if __name__ == "__main__":
    main()
