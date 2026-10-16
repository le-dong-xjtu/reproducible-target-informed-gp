# Deterministic Target-Conditioned GP Optimization

Code, configurations, row-level results, and reproducibility checks for
**“Deterministic Target-Conditioned Gaussian-Process Optimization for
Small-Budget Electromagnetic Design.”**

Public repository: https://github.com/le-dong-xjtu/reproducible-target-informed-gp

The study develops an auditable low-budget constrained Bayesian-optimization
workflow. Its core contribution is reproducible allocation of observations:
deterministic center initialization, uncertainty-aware feasibility discovery,
fixed Gaussian-process hyperparameter starts, a dimension-dependent minimum
sample threshold, relative constrained-expected-improvement stopping, and a
ten-evaluation final verification batch included in the evaluation budget. Wireless-power-
transfer (WPT) experiments test the frozen workflow on numerical and direct
field-solver models.

## Reproduced claims

- Matched GP/neural-surrogate and acquisition comparisons on a 1,110-design
  two-dimensional coil lattice.
- Frozen development/held-out protocol: 73/73 held-out exhaustive optima with
  21.81 mean model evaluations.
- Random-six versus deterministic-center adaptive-stopping evidence.
- A 21-target, eight-dimensional numerical WPT application with matched
  controls.
- A matched 1000-nH Ansys Maxwell application check: 30 accepted evaluations
  per policy, with complete evaluation histories and the selected automatic
  AEDT project.

## Quick verification

Python 3.11--3.13 is recommended. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python essay/scripts/audit_revised_manuscript.py --numerical-only
python essay/scripts/audit_aedt_wpt_matched.py
```

The first check covers the numerical experiments, budgets, held-out results,
stopping histories, deterministic replays, and WPT controls. The second checks
every claim in the direct Maxwell comparison from the archived accepted-
evaluation records and runs without Ansys.

## Repository map

- `methods/surrogate_selection/`: target-independent GP and neural-surrogate
  comparisons from the first experimental stage.
- `methods/target_informed_gp/`: the conventional cEI/cLCB/IVR acquisition
  comparison from the second stage, explicitly separated from the proposed
  automatic workflow.
- `methods/direct_search_baselines/`: self-contained exhaustive, SA, GA, and
  PSO comparison scripts; no shared optimizer engine is required.
- `experiments/`: two-dimensional benchmark runners.
- `config/`: shared physical and optimizer configurations.
- `results/coil_2d_relative_tolerance/`: retained benchmark outputs.
- `essay/scripts/run_automatic_gp_2d.py`: proposed automatic workflow on the
  held-out 2-D coil targets.
- `essay/scripts/run_automatic_gp_wpt8d.py`: proposed automatic workflow on
  the 8-D WPT benchmark.
- `essay/scripts/wpt_partial_inductance_model.py`: self-contained numerical
  WPT model used by the 8-D runners; it replaces the former omitted generic
  `core.py` dependency.
- `essay/scripts/`: remaining controls, analyses, renderers, and audits.
- `essay/data/`: row-level evidence and machine-readable summaries.
- `essay/data/aedt_wpt_1000_matched/`: complete Maxwell run archives.
- `essay/evidence_manifest.md`: claim-to-code-to-data map.
- `essay/coil_surrogate_icece_5page.pdf`: five-page conference manuscript.

## Rebuild the conference paper

A TeX distribution with `pdflatex` is required:

```sh
cd essay
pdflatex -interaction=nonstopmode -halt-on-error coil_surrogate_icece_5page.tex
pdflatex -interaction=nonstopmode -halt-on-error coil_surrogate_icece_5page.tex
```

## Regenerate figures

```sh
python essay/scripts/render_figures.py
python essay/scripts/render_automated_stopping.py
python essay/scripts/render_wpt_selected_geometry.py
```

Figure rendering uses retained data and does not rerun the optimization
sweeps.

## Maxwell evidence

The accepted-evaluation records, exact optimizers, solver-construction script, logs,
frozen confirmation plans, summaries, and selected AEDT case are archived in
`essay/data/aedt_wpt_1000_matched/`. Reproducing new physical calls requires
Ansys Electronics Desktop 2020.1 on Windows; auditing the reported records does
not. Failed solver launches repeat the identical frozen geometry and are not
counted as observations. The automatic and baseline winning designs used
different declared mesh paths, so the paper treats their 1.50% objective gap
as provisional pending a common mesh-convergence check.

## Reproducibility scope

The third-party WPT source article is cited but not redistributed. The public
package contains no credentials, private paths required for audit execution,
or superseded disconnected AEDT handoff artifacts. The retained experiments
are those needed to audit the published claims and their declared controls.

## Citation

Machine-readable citation metadata are provided in `CITATION.cff`. A fixed
Zenodo DOI can be added after the first public GitHub release is archived.

## Author

Le Dong is an undergraduate student in Electrical Engineering at Xi'an
Jiaotong University. His current research focuses on data-efficient
electromagnetic design, Gaussian-process optimization, wireless power transfer,
and reproducible computational methods for engineering applications.

## License

The code and accompanying reproducibility materials are released under the
MIT License; see `LICENSE`.
