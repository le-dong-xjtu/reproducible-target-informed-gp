# Deterministic Target-Conditioned Gaussian-Process Optimization

Code, results, figures, and manuscript source for:

> Le Dong, “Deterministic Target-Conditioned Gaussian-Process Optimization for
> Small-Budget Electromagnetic Design,” Xi'an Jiaotong University.

- [Manuscript PDF](essay/manuscript.pdf)
- [LaTeX source](essay/manuscript.tex)
- Repository: <https://github.com/le-dong-xjtu/reproducible-target-informed-gp>

## Repository layout

```text
config/                  Experiment settings
essay/                   Manuscript source and compiled PDF only
experiments/             Experiment, audit, and figure-generation programs
figures/                 Five generated PDFs and the Maxwell image used as Fig. 6
methods/                 Reusable GP and WPT model implementations
results/
  coil_2d_relative_tolerance/  Fixed-budget 2-D comparisons
  validation/                  Adaptive 2-D, WPT, and Maxwell evidence
requirements.txt         Complete Python environment
```

`figures/fig06_maxwell_selected_geometry.png` is the Ansys Maxwell screenshot
included directly as Fig. 6.
The former Fig. 4 evidence JSON was removed because the same information is
already stored in `results/validation/wpt_starting_policy_screen_uncapped/`.

## Install

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

There is one requirements file. The deep-ensemble comparison uses
scikit-learn's `MLPRegressor`; PyTorch is not required.

For deterministic numerical runs, use one thread per numerical library:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

## Verify retained results

Run from the repository root:

```sh
python3 experiments/verify_clcb_replay.py
python3 experiments/verify_uncapped_rbf_submission.py
python3 experiments/verify_uncapped_wpt_comparison.py
python3 experiments/verify_wpt_starting_policy_screen.py
python3 experiments/audit_aedt_wpt_matched.py
```

These checks cover the fixed-budget acquisition comparison, adaptive 2-D
validation, 21-target WPT comparison, WPT initialization screen, and the two
archived Maxwell runs.

## Rebuild figures and manuscript

```sh
python3 experiments/render_conference_figures.py
cd essay
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

The figure program writes `figures/fig01_*.pdf` through
`figures/fig05_*.pdf`. The manuscript includes those files and the Maxwell PNG
directly.

## Evidence locations

| Paper item | Evidence | Program |
| --- | --- | --- |
| Figs. 1–3 | `results/coil_2d_relative_tolerance/` | `experiments/run_global_*.py`, `run_cei_budget27.py`, `run_standard_bo_budget27.py` |
| Table I | `results/validation/center_stopping_*`, `random6_adaptive_stopping_uncapped_rbf/` | `run_uncapped_automatic_2d.py`, `run_random_start_adaptive_stopping_uncapped.py` |
| Table II | `results/validation/wpt_8d_multitarget_validation_uncapped/` | `run_wpt_8d_multitarget_validation.py` |
| Fig. 4 | `results/validation/wpt_starting_policy_screen_uncapped/` | `run_wpt_starting_policy_screen.py` |
| Fig. 5 | `results/validation/wpt_8d_automated_cei/` | `run_automatic_gp_wpt8d.py` |
| Fig. 6 | `results/validation/aedt_wpt_1000_matched/` | `audit_aedt_wpt_matched.py` |

## License

MIT. See [LICENSE](LICENSE).
