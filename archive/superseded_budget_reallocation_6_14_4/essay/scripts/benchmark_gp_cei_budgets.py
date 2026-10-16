"""
Comprehensive Study: GP-cEI Budget Scaling and Allocation Strategy
Evaluates GP-cEI performance on the IEEE coil optimization benchmark:
- 1,110 discrete coil designs
- 87 target inductances (1 to 87 mH)
- 10 fixed seeds (20261015 to 20261024)
- 870 evaluation cases per configuration
"""

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', ConvergenceWarning)

# Physics setup matching config/coil_problem.json
w = {'diameter_mm': 0.68, 'winding_factor': 1.05, 'resistivity_ohm_mm': 1.851e-5}
p = w['diameter_mm'] * w['winding_factor']
g = {'inner_diameter_mm': 80.0, 'layers': {'min': 1, 'max': 10}, 'turns_per_layer': {'min': 20, 'max': 130}}
pts = [(T, n) for T in range(1, 11) for n in range(20, 131)]
X = np.array([[(T - 1) / 9.0, (n - 20) / 110.0] for T, n in pts])
L_all = np.array([0.08 * (T*n)**2 * (80.0 + T*p)**2 * 1e-4 / (3 * (80.0 + T*p) + 9 * (n*p) + 10 * (T*p)) for T, n in pts])
R_all = np.array([4 * w['resistivity_ohm_mm'] * (T*n) * (80.0 + T*p) / w['diameter_mm']**2 for T, n in pts])

epsilon = 0.025
SEEDS = tuple(range(20261015, 20261025))
TARGETS = tuple(range(1, 88))

# Precompute exhaustive ground truth optima
optima = {}
for target in TARGETS:
    feas = [i for i in range(len(pts)) if abs(L_all[i] - target) / target <= epsilon]
    optima[target] = min([R_all[i] for i in feas])


def evaluate_single_seed_target(args):
    seed, target, n_init, n_active, n_val, hybrid_cover_weight = args
    rng = random.Random(seed)
    seen = set(rng.sample(range(len(pts)), n_init))

    init_ids = sorted(seen)
    scale_L = max(float(np.std(L_all[init_ids])), 1e-12)
    scale_R = max(float(np.std(R_all[init_ids])), 1e-12)

    # Active acquisition loop
    for step in range(n_active):
        ids = sorted(seen)
        gp_L = GaussianProcessRegressor(
            kernel=ConstantKernel(1., (1e-3, 1e3)) * RBF([.25, .25], (1e-2, 10.0)) + WhiteKernel(1e-6, (1e-8, 1e-2)),
            alpha=1e-10, normalize_y=True, random_state=seed, n_restarts_optimizer=2
        ).fit(X[ids], L_all[ids])
        gp_R = GaussianProcessRegressor(
            kernel=ConstantKernel(1., (1e-3, 1e3)) * RBF([.25, .25], (1e-2, 10.0)) + WhiteKernel(1e-6, (1e-8, 1e-2)),
            alpha=1e-10, normalize_y=True, random_state=seed, n_restarts_optimizer=2
        ).fit(X[ids], R_all[ids])

        unseen = np.array([i for i in range(len(pts)) if i not in seen])
        mean_L, std_L = gp_L.predict(X[unseen], return_std=True)
        mean_R, std_R = gp_R.predict(X[unseen], return_std=True)

        sigma_L = np.maximum(std_L, 1e-12)
        lower, upper = target * (1 - epsilon), target * (1 + epsilon)
        p_f = ndtr((upper - mean_L) / sigma_L) - ndtr((lower - mean_L) / sigma_L)

        feas_seen = [i for i in ids if abs(L_all[i] - target) / target <= epsilon]
        if feas_seen:
            incumbent = min(R_all[i] for i in feas_seen)
            sigma_R = np.maximum(std_R, 1e-12)
            z = (incumbent - mean_R) / sigma_R
            ei = (incumbent - mean_R) * ndtr(z) + sigma_R * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            acq = p_f * np.maximum(ei, 0) / incumbent
        else:
            acq = p_f

        if hybrid_cover_weight > 0.0:
            span_acq = np.ptp(acq)
            norm_acq = (acq - np.min(acq)) / span_acq if span_acq > 0 else np.zeros_like(acq)
            cov = 0.5 * sigma_L / scale_L + 0.5 * np.maximum(std_R, 1e-12) / scale_R
            span_cov = np.ptp(cov)
            norm_cov = (cov - np.min(cov)) / span_cov if span_cov > 0 else np.zeros_like(cov)
            acq = (1 - hybrid_cover_weight) * norm_acq + hybrid_cover_weight * norm_cov

        selected = unseen[int(np.argmax(acq))]
        seen.add(selected)

    ids = sorted(seen)
    if n_val > 0:
        gp_L = GaussianProcessRegressor(
            kernel=ConstantKernel(1., (1e-3, 1e3)) * RBF([.25, .25], (1e-2, 10.0)) + WhiteKernel(1e-6, (1e-8, 1e-2)),
            alpha=1e-10, normalize_y=True, random_state=seed, n_restarts_optimizer=2
        ).fit(X[ids], L_all[ids])
        gp_R = GaussianProcessRegressor(
            kernel=ConstantKernel(1., (1e-3, 1e3)) * RBF([.25, .25], (1e-2, 10.0)) + WhiteKernel(1e-6, (1e-8, 1e-2)),
            alpha=1e-10, normalize_y=True, random_state=seed, n_restarts_optimizer=2
        ).fit(X[ids], R_all[ids])

        unseen = np.array([i for i in range(len(pts)) if i not in seen])
        mean_L, std_L = gp_L.predict(X[unseen], return_std=True)
        mean_R, std_R = gp_R.predict(X[unseen], return_std=True)
        sigma_L = np.maximum(std_L, 1e-12)
        lower, upper = target * (1 - epsilon), target * (1 + epsilon)
        p_f = ndtr((upper - mean_L) / sigma_L) - ndtr((lower - mean_L) / sigma_L)
        span_R = np.ptp(mean_R)
        scaled_R = (mean_R - np.min(mean_R)) / span_R if span_R > 0 else np.zeros_like(mean_R)
        val_score = p_f * np.exp(-scaled_R)
        val_ids = unseen[np.argsort(-val_score)[:n_val]]
        eval_ids = ids + list(val_ids)
    else:
        eval_ids = ids

    feas_eval = [i for i in eval_ids if abs(L_all[i] - target) / target <= epsilon]
    if feas_eval:
        dec = min(feas_eval, key=lambda i: R_all[i])
        opt_R = optima[target]
        regret = R_all[dec] / opt_R - 1.0
        loss = regret / (1.0 + regret)
        is_exact = regret <= 1e-12
        is_qual = True
    else:
        loss = 1.0
        is_exact = False
        is_qual = False

    return {
        'seed': seed,
        'target': target,
        'qualified': is_qual,
        'exact': is_exact,
        'loss': loss
    }


def evaluate_configuration(config_name, n_init, n_active, n_val, hybrid_cover_weight=0.0, workers=10):
    total_budget = n_init + n_active + n_val
    tasks = [(seed, target, n_init, n_active, n_val, hybrid_cover_weight)
             for seed in SEEDS for target in TARGETS]
    
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(evaluate_single_seed_target, tasks))
    elapsed = time.time() - t0

    total_cases = len(results)
    qual_count = sum(r['qualified'] for r in results)
    exact_count = sum(r['exact'] for r in results)
    losses = [r['loss'] for r in results]

    # Targetwise trimmed mean (remove min and max of the 10 seeds at each target)
    by_target = {t: [] for t in TARGETS}
    for r in results:
        by_target[r['target']].append(r['loss'])
    trimmed_target_means = [sum(sorted(by_target[t])[1:-1]) / 8.0 for t in TARGETS]
    trimmed_mean = float(np.mean(trimmed_target_means))

    # Worst seed mean
    by_seed = {s: [] for s in SEEDS}
    for r in results:
        by_seed[r['seed']].append(r['loss'])
    seed_means = {s: float(np.mean(by_seed[s])) for s in SEEDS}
    worst_seed = max(seed_means, key=seed_means.get)
    worst_seed_loss = seed_means[worst_seed]

    summary = {
        'config': config_name,
        'budget': total_budget,
        'n_init': n_init,
        'n_active': n_active,
        'n_val': n_val,
        'cover_weight': hybrid_cover_weight,
        'total_cases': total_cases,
        'qualification_rate': qual_count / total_cases,
        'exact_rate': exact_count / total_cases,
        'exact_count': exact_count,
        'all_case_mean_loss': float(np.mean(losses)),
        'trimmed_mean_loss': trimmed_mean,
        'worst_seed_mean_loss': worst_seed_loss,
        'worst_seed': worst_seed,
        'runtime_s': elapsed
    }
    return summary


if __name__ == '__main__':
    configs = [
        ("Paper Baseline: Pure cEI (B=27)", 6, 9, 12, 0.0),
        ("Paper Hybrid: 75% cEI + 25% Cov (B=27)", 6, 9, 12, 0.25),
        ("Active-Boosted: B=24 (init=6, active=14, val=4, pure cEI)", 6, 14, 4, 0.0),
        ("Active-Boosted: B=21 (init=6, active=12, val=3, pure cEI)", 6, 12, 3, 0.0),
        ("Active-Boosted: B=20 (init=6, active=11, val=3, pure cEI)", 6, 11, 3, 0.0),
        ("Active-Boosted: B=18 (init=6, active=10, val=2, pure cEI)", 6, 10, 2, 0.0),
        ("Zero Val: B=15 (init=6, active=9, val=0)", 6, 9, 0, 0.0),
    ]

    print("=" * 80)
    print("RUNNING GP-cEI SYSTEMATIC BUDGET SCALING BENCHMARK (10 SEEDS x 87 TARGETS)")
    print("=" * 80)

    summaries = []
    for cfg in configs:
        res = evaluate_configuration(*cfg)
        summaries.append(res)
        print(f"[{res['config']}]")
        print(f"  Budget: {res['budget']:2d} | Qual: {res['qualification_rate']*100:.2f}% | "
              f"Exact: {res['exact_count']}/870 ({res['exact_rate']*100:.2f}%) | "
              f"Mean Loss: {res['all_case_mean_loss']:.6f} | Trimmed: {res['trimmed_mean_loss']:.6f} | "
              f"Worst Seed: {res['worst_seed_mean_loss']:.6f} | Time: {res['runtime_s']:.1f}s")
        print("-" * 80)

    out_json = Path(__file__).resolve().parent.parent / "data" / "gp_cei_budget_scaling_summary.json"
    out_json.write_text(json.dumps(summaries, indent=2))
    print(f"\nAll results saved to: {out_json}")
