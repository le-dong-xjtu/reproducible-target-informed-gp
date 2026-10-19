# Active evidence map

All paths are relative to this package. `essay/scripts/render_conference_figures.py` generates exactly the six PDFs included by the manuscript in `essay/figures/conference/`.

| Paper component | Active raw evidence | Generator / check |
|---|---|---|
| Fig. 1: GP/ensemble | results/coil_2d_relative_tolerance/surrogate_family_comparison_budget27/results.csv | experiments/run_global_surrogate_2x2_budget27.py |
| Fig. 2: GP/SA | same GP records; essay/data/direct_optimizer_summary_trimmed.csv; results/coil_2d_relative_tolerance/direct_optimizer_baselines | experiments/adaptive_relative_tolerance_direct.py; unified renderer |
| Fig. 3: acquisition | results/coil_2d_relative_tolerance/global_dual_gp_budget27; cei_budget27; standard_bo_budget27 | experiments/run_global_dual_gp_budget27.py; run_cei_budget27.py; run_standard_bo_budget27.py |
| Table I: center | essay/data/center_stopping_1pct_deterministic_gp_k5; center_stopping_heldout_73 | run_uncapped_automatic_2d.py --policy center; verify_uncapped_rbf_submission.py |
| Table I: random-six | essay/data/random6_adaptive_stopping_uncapped_rbf | run_random_start_adaptive_stopping_uncapped.py; verify_uncapped_rbf_submission.py |
| Local/global calibration | essay/data/stopping_gp_calibration_heldout_73 | analyze_stopping_gp_calibration.py |
| Table II: 21 WPT targets | essay/data/wpt_8d_multitarget_validation_uncapped | run_wpt_8d_multitarget_validation.py; verify_uncapped_wpt_comparison.py |
| Fig. 4: selected random seed / center | essay/data/wpt_starting_policy_screen_uncapped | run_wpt_starting_policy_screen.py; verify_wpt_starting_policy_screen.py |
| Fig. 5: numerical geometry | selected 1500-nH design in the WPT records | unified renderer |
| Maxwell results / Fig. 6 | essay/data/aedt_wpt_1000_matched; essay/figures/fig07_maxwell_selected_geometry.png | scripts inside original ZIPs; audit_aedt_wpt_matched.py; unified renderer |

## Model configuration locations

- Original 2-D automatic RBF definition, bounds, normalization, alpha and deterministic starts: essay/scripts/run_center_start_control.py.
- Global GP/ensemble configuration: experiments/run_global_surrogate_2x2_budget27.py.
- Fixed-budget acquisition definitions: methods/target_informed_gp/gp_conventional_acquisition_baselines.py.
- Numerical WPT Matérn configuration and candidate construction: essay/scripts/run_automatic_gp_wpt8d.py.
- Maxwell configuration: original scripts in essay/data/aedt_wpt_1000_matched/automatic_exact_run.zip and sobol6_cei14_baseline.zip.

## Notes

Reproducibility checks verify records and their numerical consistency. They do not certify physical model accuracy, mesh convergence, hardware performance, or conference compliance.
