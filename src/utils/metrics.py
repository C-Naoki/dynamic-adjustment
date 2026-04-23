from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig
from scipy.stats import norm

from src.models.dynamicra.model import EstimateResult


def calculate_metrics(
    results: List[List[EstimateResult]],
    truth: Optional[Dict[str, Any]],
    cfg: DictConfig,
) -> Dict[str, Any]:
    # Collect all unique raw time indices observed across iterations (0-based).
    iterations = len(results)
    raw_times = sorted({res['time'] for iter_res in results for res in iter_res if 'time' in res})
    path_tuple: Tuple[int, ...] = ()
    control_tuple: Optional[Tuple[int, ...]] = None
    if cfg is not None and 'model' in cfg and cfg['model'] is not None:
        model_cfg = cfg['model']
        if 'treatment_path' in model_cfg and model_cfg['treatment_path'] is not None:
            path_tuple = tuple(model_cfg['treatment_path'])
        if 'control_path' in model_cfg and model_cfg.get('control_path') is not None:
            control_tuple = tuple(model_cfg['control_path'])

    is_ate = any(('mu_treatment' in rec) or ('mu_control' in rec) for iter_res in results for rec in iter_res)

    # Fetch the ground truth trajectory (0-based array indexed by t).
    truth_path = None
    if truth is not None:
        mu_by_path = truth.get('mu_by_path', {})
        if is_ate and control_tuple is not None:
            treat_truth = mu_by_path.get(path_tuple)
            control_truth = mu_by_path.get(control_tuple)
            if treat_truth is not None and control_truth is not None:
                truth_path = np.asarray(treat_truth, dtype=float) - np.asarray(control_truth, dtype=float)
        else:
            truth_path = mu_by_path.get(path_tuple)
            if truth_path is not None:
                truth_path = np.asarray(truth_path, dtype=float)

    time_values = raw_times

    # Extract estimates and CIs into arrays of shape (iterations, T)
    T = len(time_values)
    mu_hat = np.full((iterations, T), np.nan, dtype=float)
    ci_lower = np.full((iterations, T), np.nan, dtype=float)
    ci_upper = np.full((iterations, T), np.nan, dtype=float)
    baseline = np.full((iterations, T), np.nan, dtype=float)
    residual = np.full((iterations, T), np.nan, dtype=float)
    transition = np.full((iterations, T), np.nan, dtype=float)
    n_path = np.full((iterations, T), np.nan, dtype=float)
    n_total = np.full((iterations, T), np.nan, dtype=float)
    support_ratio = np.full((iterations, T), np.nan, dtype=float)
    for it, iter_results in enumerate(results):
        by_time = {rec['time']: rec for rec in iter_results}
        for col, t_val in enumerate(time_values):
            mu_hat[it, col] = by_time[t_val]['mu']
            ci_lower[it, col] = by_time[t_val]['ci_lower']
            ci_upper[it, col] = by_time[t_val]['ci_upper']
            baseline[it, col] = by_time[t_val]['baseline']
            residual[it, col] = by_time[t_val]['residual']
            transition[it, col] = by_time[t_val]['trans_aug_mean']
            n_path_val = by_time[t_val]['n_path']
            n_total_val = by_time[t_val]['n_total']
            n_path[it, col] = n_path_val
            n_total[it, col] = n_total_val
            if np.isfinite(n_path_val) and np.isfinite(n_total_val) and n_total_val > 0.0:
                support_ratio[it, col] = n_path_val / n_total_val

    truth_series = np.full((T,), np.nan, dtype=float)
    if truth_path is not None:
        for idx, t_val in enumerate(time_values):
            if 0 <= t_val < truth_path.shape[0]:
                truth_series[idx] = truth_path[t_val]

    # Confidence level
    ci_level = cfg.eval.ci_level
    assert 0 < ci_level < 1.0, 'ci_level must be in (0,1).'
    alpha = 1.0 - ci_level
    z = norm.ppf(1.0 - alpha / 2.0)
    per_time: List[Dict[str, Any]] = []
    biases = []
    rmses = []
    for col, t_val in enumerate(time_values):
        mu_vals = mu_hat[:, col]
        truth_val = truth_series[col]
        valid_mu = ~np.isnan(mu_vals)
        n_valid = valid_mu.sum()

        if n_valid == 0:
            raise ValueError(f'No valid estimates available to compute metrics for t={t_val}.')
        mu_clean = mu_vals[valid_mu]

        mean_hat = np.mean(mu_clean)
        std_hat = np.std(mu_clean, ddof=1)
        bias = mean_hat - truth_val
        rmse = np.sqrt(np.mean((mu_clean - truth_val) ** 2))
        empirical_ci_low = np.percentile(mu_clean, alpha / 2 * 100)
        empirical_ci_high = np.percentile(mu_clean, (1 - alpha / 2) * 100)
        ci_lo_vals = ci_lower[:, col]
        ci_hi_vals = ci_upper[:, col]
        valid_ci = np.isfinite(ci_lo_vals) & np.isfinite(ci_hi_vals)

        boot_lo = np.mean(ci_lo_vals[valid_ci])
        boot_hi = np.mean(ci_hi_vals[valid_ci])
        se_vals = (ci_hi_vals[valid_ci] - ci_lo_vals[valid_ci]) / (2.0 * z)
        boot_se = np.mean(se_vals)
        coverage = np.mean((ci_lo_vals[valid_ci] <= truth_val) & (truth_val <= ci_hi_vals[valid_ci]))

        baseline_vals = baseline[:, col]
        residual_vals = residual[:, col]
        transition_vals = transition[:, col]
        n_path_vals = n_path[:, col]
        n_total_vals = n_total[:, col]
        support_vals = support_ratio[:, col]

        baseline_mean = np.mean(baseline_vals)
        residual_mean = np.mean(residual_vals)
        transition_mean = np.mean(transition_vals)

        # Dynamic minus baseline evaluated where both finite
        delta_vals = np.where(
            np.isfinite(mu_vals) & np.isfinite(baseline_vals),
            mu_vals - baseline_vals,
            np.nan,
        )
        delta_mean = np.mean(delta_vals)

        n_path_mean = np.mean(n_path_vals)
        n_total_mean = np.mean(n_total_vals)
        support_mean = np.mean(support_vals)
        support_std = np.std(support_vals)

        per_time.append(
            {
                't': t_val,
                'truth': truth_val,
                'mean_hat': mean_hat,
                'std_hat': std_hat,
                'bias': bias,
                'rmse': rmse,
                'empirical_ci_low': empirical_ci_low,
                'empirical_ci_high': empirical_ci_high,
                'bootstrap_mean_ci_low': boot_lo,
                'bootstrap_mean_ci_high': boot_hi,
                'bootstrap_mean_se': boot_se,
                'bootstrap_coverage': coverage,
                'baseline_mean': baseline_mean,
                'residual_mean': residual_mean,
                'trans_aug_mean': transition_mean,
                'delta_mean': delta_mean,
                'n_path_mean': n_path_mean,
                'n_total_mean': n_total_mean,
                'support_ratio_mean': support_mean,
                'support_ratio_std': support_std,
            }
        )
        biases.append(bias)
        rmses.append(rmse)

    overall = {
        'rmse_mean': np.mean(rmses),
        'abs_bias_mean': np.mean(np.abs(biases)),
    }

    return {
        'iterations': iterations,
        'T': T,
        'times': time_values,
        'treatment_path': path_tuple,
        'control_path': control_tuple,
        'estimand': 'ate' if is_ate else 'path_mean',
        'ci_level': ci_level,
        'per_time': per_time,
        'overall': overall,
    }
