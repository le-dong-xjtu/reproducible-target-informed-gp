# Matched 1000-nH Maxwell evidence

These immutable archives contain the two direct Ansys Maxwell trajectories
reported in the full manuscript.

- `automatic_exact_run.zip`: center-start automatic workflow, including the
  complete source/ledger package and `best_call_025_case.aedt`.
- `sobol6_cei14_baseline.zip`: six fixed Sobol initial evaluations, fourteen
  conventional-cEI evaluations, and ten frozen confirmations.

Both runs use 20 accepted model-building observations and 10 accepted
confirmations. Failed solver attempts repeat the identical frozen geometry and
are not counted as observations. Run the independent claim audit with:

```sh
python3 experiments/audit_aedt_wpt_matched.py
```

SHA-256:

```text
6b58efe2471c9a3b08e1b5b0d1d696abf1cbb273e674e992f8025e270f1f1a43  automatic_exact_run.zip
4fb4225c0f02ad963545ac92812b1729e20b0917aad5cafc59c66af6e1c5af84  sobol6_cei14_baseline.zip
```

The automatic winner used the declared 1-mm copper-mesh fallback, whereas the
baseline winner completed with the default adaptive mesh. The archives support
the direct-execution and accepted-budget claims; a common mesh-convergence
rerun is still required before treating the 1.50% objective difference as a
definitive high-fidelity performance margin.
