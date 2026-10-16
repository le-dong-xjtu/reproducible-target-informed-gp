# Reproducibility and evidence manifest

This file maps the empirical figures, tables, and numerical studies in the
complete study, including the conference subset in
`essay/coil_surrogate_icece_5page.tex`, to generating code and retained data.
Paths are relative to the repository root unless stated otherwise.

## Shared definitions

- Two-dimensional coil oracle and configuration:
  `config/coil_problem.json` and the oracle implementations in the experiment
  scripts listed below.
- Bounded loss used by every 2-D experiment:
  `r = R / R_exhaustive - 1`, `u = r / (1 + r)` for a qualified design, and
  `u = 1` for an unqualified design.
- Eight-dimensional source model: the third-party article PDF
  `elp2.70015.pdf` is intentionally not redistributed; its public citation and
  DOI appear in the manuscript. The precise 14-to-8 interpretation is in
  `essay/wpt_source_scope.md`.

## Figure and table map

| Paper item | Experiment code | Raw data / summary | Rendering or analysis code |
|---|---|---|---|
| Fig. 1, matched GP/ensemble comparison | `experiments/run_global_surrogate_2x2_budget27.py` | `results/coil_2d_relative_tolerance/surrogate_family_comparison_budget27/results.csv`, `summary.json`, `manifest.json` | `essay/scripts/render_figures.py` |
| Fig. 2, direct optimizers | `experiments/adaptive_relative_tolerance_direct.py` | `results/coil_2d_relative_tolerance/direct_optimizer_baselines/results.csv`, plus `essay/data/direct_optimizer_results_raw_10seeds.csv` and `direct_optimizer_summary_trimmed.csv` | `essay/scripts/render_figures.py` |
| Fig. 3, global GP versus SA | `experiments/run_global_dual_gp_budget27.py`; SA code above | `results/coil_2d_relative_tolerance/global_dual_gp_budget27/alpha_0p5/results.csv`; direct-optimizer data above | `essay/scripts/render_figures.py` |
| Fig. 4, GP acquisitions | `experiments/run_cei_budget27.py`, `experiments/run_standard_bo_budget27.py`, `experiments/run_global_dual_gp_budget27.py` | `results/coil_2d_relative_tolerance/cei_budget27/cei_results.csv`; `standard_bo_budget27/clcb_wl1_wg0_a0p5/results.csv`; `standard_bo_budget27/ivr_wl1_wg0_a0p5/results.csv`; global-GP data above | `essay/scripts/render_figures.py` |
| Correct-sign cLCB replay audit | `essay/scripts/verify_clcb_replay.py`; acquisition implementation in `methods/target_informed_gp/gp_conventional_acquisition_baselines.py` | `essay/data/clcb_correct_sign_replay/results.csv`, `verification.json` | exact rowwise comparison with the archived Fig. 4 input |
| Reusable-GP calibration diagnostic | `experiments/analyze_gp_errors_budget27.py` | `essay/data/gp_budget_27_grid_error_by_seed.csv`, `gp_budget_27_error_analysis.json` | same analysis script; summary reported in text |
| Ten-target feasibility-weight screen | `essay/scripts/run_center_feasibility_smoke.py` | `essay/data/center_feasibility_smoke/results.csv`, `summary.json`, `manifest.json` | analysis embedded in the experiment script |
| Post-hoc $L_t=2$--5 mH initialization diagnostic | `essay/scripts/run_center_feasibility_low_targets.py` | `essay/data/center_feasibility_low_targets/results.csv`, `summary.json`, `manifest.json` | analysis embedded in the experiment script |
| Prespecified 1,000-seed initial-coverage screen | `essay/scripts/seed_initialization_screen.py` | `essay/data/initialization_screen_1000/seed_initialization_1000x87.csv`, `seed_initialization_points.csv`, `targetwise_initialization_summary.csv`, `screening_summary.json` | `essay/scripts/verify_initialization_screen.py`; initialization only, not GP-cEI trajectories |
| Preselected worst-initialization end-to-end illustration | `essay/scripts/run_worst_initialization_case.py` | `essay/data/initialization_screen_1000/worst_seed_end_to_end/result.json` | injects the archived six designs for seed 20261105; both methods recover the exact 2-mH optimum, at 27 versus 23 calls |
| Random-six versus center adaptive-stopping comparison | `essay/scripts/run_random_start_adaptive_stopping.py` | `essay/data/random6_adaptive_stopping/results.csv`, `summary.json`, `manifest.json` | all 87 targets and ten seeds; common discovery, deterministic GP, stopping, confirmation, and maximum-budget rules |
| Fig. 5(a), stopping and deterministic-GP screen | `essay/scripts/run_center_cosine_stopping_smoke.py`; deterministic optimizer in `run_center_start_control.py` | `essay/data/center_cosine_stopping_smoke/`, `center_stopping_1pct_smoke/`, `center_stopping_1pct_deterministic_gp_k5/` | `essay/scripts/render_automated_stopping.py` |
| Frozen stopping test on 73 held-out targets | `essay/scripts/run_automatic_gp_2d.py` | `essay/data/center_stopping_heldout_73/results.csv`, `summary.json`, `manifest.json` | analysis embedded in the experiment script; independently checked by the audit |
| Target-conditioned GP calibration exactly at stopping | `essay/scripts/analyze_stopping_gp_calibration.py` | `essay/data/stopping_gp_calibration_heldout_73/results.csv`, `summary.json`, `manifest.json` | reports global and true-feasible-region coverage before confirmations |
| Fig. 5(b), automated 8-D WPT trace | `essay/scripts/run_automatic_gp_wpt8d.py` | `essay/data/wpt_8d_automated_cei/`; forced-budget control in `wpt_8d_automated_cei_fixed_b50_control/` | `essay/scripts/render_automated_stopping.py` |
| Fig. 6, selected WPT geometry | `essay/scripts/run_automatic_gp_wpt8d.py` | `essay/data/wpt_8d_automated_cei/result.json` | `essay/scripts/render_wpt_selected_geometry.py` |
| Frozen 21-target matched 8-D WPT sweep and representative table | `essay/scripts/run_wpt_8d_multitarget_validation.py` | `essay/data/wpt_8d_multitarget_validation/results.csv`, `summary.json`, `manifest.json` | automatic and conventional controls use deterministic Sobol candidate streams; the audit recomputes the paired Wilcoxon result and the four 17-observation guard-boundary cases |
| Development-only confirmation-count ablation | `essay/scripts/analyze_confirmation_ablation.py` | `essay/data/confirmation_count_ablation/results.csv`, `summary.json` | replays prefixes of stored ranked confirmations without refitting or retuning |
| Earlier ten-seed 8-D pure-cEI context | `essay/scripts/run_wpt_8d_mixed_benchmark.py` | `essay/data/wpt_8d_mixed/results.csv`, `summary.json`, `manifest.json` | summary reported in text |
| Resistance-capped application study | `essay/scripts/run_wpt_8d_resistance_constrained.py` | `essay/data/wpt_8d_resistance_constrained/results.csv`, `summary.json`, `manifest.json` | summary reported in text |
| WPT budget sweep over every integer $B=30,\ldots,80$ | `essay/scripts/run_wpt_8d_budget_sweep.py` | `essay/data/wpt_8d_budget_sweep_b30_b80/results.csv`, `summary.json`, `manifest.json` | `essay/scripts/render_wpt_budget_sweep.py`; summary reported in text |
| Matched 1000-nH direct Maxwell execution | archived exact runners in `essay/data/aedt_wpt_1000_matched/automatic_exact_run.zip` and `sobol6_cei14_baseline.zip` | complete trajectory CSV/JSON, frozen confirmation plans, logs, summaries, and selected automatic AEDT project inside the archives | `essay/scripts/audit_aedt_wpt_matched.py`; checks budgets, phase counts, uniqueness, feasibility, selected responses, resistance difference, natural stop, and mesh caveat |

The executable integrity check is
`essay/scripts/audit_revised_manuscript.py`. It independently recomputes every
reported 2-D bounded loss, verifies matched initial designs and budget
identities, reproduces the stopping summaries and deterministic seed
invariance, verifies all 73 held-out stopping trajectories, checks the
automated WPT replay and its forced-$B=50$ control, validates the 21-target
matched budgets, qualification counts, paired Wilcoxon statistic, and evidence-guard boundary count,
reproduces the retained WPT summaries and nested B=30--80 sweep, and confirms
the 87,000-row initialization screen from its archived six-point designs, and
confirms that the required manuscript assets exist.

## Build commands

From the repository root:

```sh
python3 essay/scripts/render_figures.py
python3 essay/scripts/render_automated_stopping.py
python3 essay/scripts/render_wpt_selected_geometry.py
python3 essay/scripts/render_wpt_budget_sweep.py
python3 essay/scripts/run_automatic_gp_2d.py
python3 essay/scripts/analyze_stopping_gp_calibration.py
python3 essay/scripts/analyze_confirmation_ablation.py
python3 essay/scripts/verify_initialization_screen.py
python3 essay/scripts/run_wpt_8d_multitarget_validation.py
python3 essay/scripts/verify_clcb_replay.py
python3 essay/scripts/audit_aedt_wpt_matched.py
python3 essay/scripts/audit_revised_manuscript.py
cd essay
pdflatex -interaction=nonstopmode -halt-on-error coil_surrogate_icece_5page.tex
pdflatex -interaction=nonstopmode -halt-on-error coil_surrogate_icece_5page.tex
```

Historical 14-D pose/tolerance artifacts, the earlier two-budget hybrid
experiment, the 12-verification benchmarks, and the $6+14+4$ allocation study
remain archived for provenance but are explicitly excluded from the revised
paper.
