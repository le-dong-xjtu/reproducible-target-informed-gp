# Deterministic Target-Conditioned Gaussian-Process Optimization for Small-Budget Electromagnetic Design

Companion code, data, validation scripts, and LaTeX source for the five-page conference paper:

> **Le Dong**, *"Deterministic Target-Conditioned Gaussian-Process Optimization for Small-Budget Electromagnetic Design,"* School of Electrical Engineering, Xi'an Jiaotong University.
>
> **Manuscript:** [`essay/coil_surrogate_icece_5page.pdf`](essay/coil_surrogate_icece_5page.pdf) ([LaTeX source](essay/coil_surrogate_icece_5page.tex))  
> **Repository:** https://github.com/le-dong-xjtu/reproducible-target-informed-gp

---

## 1. Overview

Electromagnetic component design frequently requires matching a target physical response within a strict tolerance band ($\pm 2.5\%$) while minimizing winding resistance or loss under a tight evaluation budget. This repository implements and validates a **deterministic, target-conditioned Gaussian-process (GP) workflow** featuring:

1. **Deterministic Initialization:** Domain-center initialization paired with an uncertainty-weighted feasibility acquisition rule ($\omega = 0.75$) prior to first feasibility.
2. **Deterministic GP Fitting:** Fixed hyperparameter warmup for the first three observations followed by deterministic Halton multistart ($K = 5$) L-BFGS-B marginal-likelihood optimization (RBF kernels in 2-D; isotropic Matérn-$5/2$ kernels in 8-D WPT).
3. **Adaptive Relative-cEI Stopping & Final Verification:** Sequential constrained expected improvement (cEI) with a dimension-dependent minimum sample threshold ($N_{\min} = 2d + 1$), relative-cEI stopping ($\rho_k \le \delta = 0.01$ over $P = 3$ consecutive refits), and a ten-candidate final verification batch included in the total evaluation budget.

---

## 2. Paper-to-Repository Index (Figures, Tables & Sections)

Every figure, table, and numerical claim in [`essay/coil_surrogate_icece_5page.pdf`](essay/coil_surrogate_icece_5page.pdf) maps directly to the scripts and datasets indexed below:

| Paper Component | Section | Active Data / Evidence Path | Generation / Runner Script | Verification / Audit Script |
| :--- | :--- | :--- | :--- | :--- |
| **Fig. 1:** Reusable GP vs. Deep Ensemble ($B = 27$) | Sec. II-B | `results/coil_2d_relative_tolerance/surrogate_family_comparison_budget27/results.csv` | `experiments/run_global_surrogate_2x2_budget27.py` | `essay/scripts/render_conference_figures.py --figures 1` |
| **Fig. 2:** Reusable Active GP vs. Simulated Annealing | Sec. II-C | `essay/data/direct_optimizer_summary_trimmed.csv`<br>`results/coil_2d_relative_tolerance/direct_optimizer_baselines/` | `experiments/adaptive_relative_tolerance_direct.py` | `essay/scripts/render_conference_figures.py --figures 2` |
| **Fig. 3:** Target-Conditioned Acquisition (`cEI`, `cLCB`, `IVR`) | Sec. II-D | `results/coil_2d_relative_tolerance/global_dual_gp_budget27/`<br>`results/coil_2d_relative_tolerance/cei_budget27/`<br>`results/coil_2d_relative_tolerance/standard_bo_budget27/` | `experiments/run_global_dual_gp_budget27.py`<br>`experiments/run_cei_budget27.py`<br>`experiments/run_standard_bo_budget27.py` | `essay/scripts/verify_clcb_replay.py`<br>`essay/scripts/render_conference_figures.py --figures 3` |
| **14-Target Development Screening** ($\omega, K, \delta, P$) | Sec. III | `essay/data/center_feasibility_smoke/`<br>`essay/data/center_feasibility_low_targets/` | `essay/scripts/run_center_feasibility_smoke.py`<br>`essay/scripts/run_center_feasibility_low_targets.py` | `essay/scripts/verify_uncapped_rbf_submission.py` |
| **Table I:** 2-D Air-Core Coil Validation (73 Held-Out & 870 Random-Six) | Sec. IV | `essay/data/center_stopping_1pct_deterministic_gp_k5/`<br>`essay/data/center_stopping_heldout_73/`<br>`essay/data/uncapped_automatic_2d_rbf_replay/`<br>`essay/data/random6_adaptive_stopping_uncapped_rbf/` | `essay/scripts/run_uncapped_automatic_2d.py`<br>`essay/scripts/run_random_start_adaptive_stopping_uncapped.py` | `essay/scripts/verify_uncapped_rbf_submission.py` |
| **Posterior Coverage Calibration** ($1\sigma / 2\sigma$ in Feasible Region) | Sec. IV | `essay/data/stopping_gp_calibration_heldout_73/` | `essay/scripts/analyze_stopping_gp_calibration.py` | `essay/data/stopping_gp_calibration_heldout_73/summary.json` |
| **Table II:** 8-D WPT Benchmark Across 21 Targets ($1000\text{--}2000\text{ nH}$) | Sec. V-B | `essay/data/wpt_8d_multitarget_validation_uncapped/` | `essay/scripts/run_wpt_8d_multitarget_validation.py` | `essay/scripts/verify_uncapped_wpt_comparison.py` |
| **Fig. 4:** 8-D WPT Initialization Trajectories at $1500\text{ nH}$ | Sec. V-B | `essay/data/wpt_starting_policy_screen_uncapped/` | `essay/scripts/run_wpt_starting_policy_screen.py` | `essay/scripts/verify_wpt_starting_policy_screen.py`<br>`essay/scripts/render_conference_figures.py --figures 4` |
| **Fig. 5:** Selected Nominal-Pose $1500\text{ nH}$ WPT Coil Geometry | Sec. V-B | `essay/data/wpt_8d_automated_cei/result.json`<br>`essay/data/wpt_8d_automated_cei_fixed_b50_control/` | `essay/scripts/run_automatic_gp_wpt8d.py` | `essay/scripts/render_conference_figures.py --figures 5` |
| **Fig. 6 & Maxwell 3-D Check:** $85\text{ kHz}$ Eddy-Current Solver at $1000\text{ nH}$ | Sec. V-C | `essay/data/aedt_wpt_1000_matched/`<br>`essay/figures/fig07_maxwell_selected_geometry.png` | Archived in `automatic_exact_run.zip` & `sobol6_cei14_baseline.zip` | `essay/scripts/audit_aedt_wpt_matched.py`<br>`essay/scripts/render_conference_figures.py --figures 6` |

---

## 3. Repository Directory Structure

```text
.
├── CITATION.cff                                  # Citation metadata
├── LICENSE                                       # MIT License
├── README.md                                     # Main repository index and usage guide
├── EVIDENCE_MANIFEST.md                          # Concise evidence mapping table
├── requirements.txt                              # Full Python dependencies
├── requirements-audit.txt                        # Minimal dependencies for record audits
├── config/
│   ├── coil_problem.json                         # 2-D air-core coil physical constants and bounds
│   └── surrogates/                               # Surrogate and acquisition JSON configs
├── methods/
│   ├── surrogate_selection/                      # Reusable dual-response GP model
│   └── target_informed_gp/                       # Conventional GP-cEI, GP-cLCB, and GP-IVR
├── experiments/                                  # 2-D benchmark runners (Figs. 1–3)
├── results/
│   └── coil_2d_relative_tolerance/               # Retained 2-D benchmark CSVs and summaries
└── essay/
    ├── coil_surrogate_icece_5page.tex            # 5-page IEEEtran conference LaTeX source
    ├── coil_surrogate_icece_5page.pdf            # Compiled 5-page conference paper
    ├── data/                                     # Row-level optimization traces and AEDT archives
    ├── figures/
    │   ├── fig07_maxwell_selected_geometry.png   # Ansys Maxwell 3-D selected geometry viewport
    │   └── conference/                           # Rendered vector PDFs (fig01–fig06)
    └── scripts/                                  # Automatic workflow runners, WPT model, audits, and renderer
```

---

## 4. Quick Verification & Rebuilding the Paper

### 4.1 Environment Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

### 4.2 Run All Numerical & Maxwell Record Audits

These checks verify all retained 2-D, 8-D WPT, and 3-D Ansys Maxwell records and regenerate all six conference figures without requiring Ansys Electronics Desktop:

```sh
python3 essay/scripts/verify_clcb_replay.py
python3 essay/scripts/verify_uncapped_rbf_submission.py
python3 essay/scripts/verify_uncapped_wpt_comparison.py
python3 essay/scripts/verify_wpt_starting_policy_screen.py
python3 essay/scripts/audit_aedt_wpt_matched.py
python3 essay/scripts/render_conference_figures.py
```

### 4.3 Compile the 5-Page IEEE Conference PDF

```sh
cd essay
pdflatex -interaction=nonstopmode -halt-on-error coil_surrogate_icece_5page.tex
pdflatex -interaction=nonstopmode -halt-on-error coil_surrogate_icece_5page.tex
```

---

## 5. Fresh Numerical Reruns

To re-execute the numerical optimization experiments from scratch (move existing output directories aside first if you do not wish to resume from cached rows):

```sh
python3 experiments/run_global_surrogate_2x2_budget27.py --workers 2
python3 experiments/run_global_dual_gp_budget27.py --workers 2
python3 experiments/run_cei_budget27.py --workers 2
python3 experiments/run_standard_bo_budget27.py --mode clcb --workers 2
python3 experiments/run_standard_bo_budget27.py --mode ivr --workers 2
python3 essay/scripts/run_uncapped_automatic_2d.py --policy center --workers 2
python3 essay/scripts/run_random_start_adaptive_stopping_uncapped.py --workers 2
python3 essay/scripts/analyze_stopping_gp_calibration.py
python3 essay/scripts/run_wpt_8d_multitarget_validation.py
python3 essay/scripts/run_wpt_starting_policy_screen.py
```

---

## 6. License

Released under the [MIT License](LICENSE).
