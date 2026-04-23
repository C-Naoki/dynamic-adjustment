from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.models.dynamicra.model import EstimateResult
from src.utils import build_arrays
from src.utils.inference import compute_ci


def run(data: pd.DataFrame, cfg: DictConfig) -> Dict[str, Any]:
    results_per_time = run_single(data, cfg)
    full_results = [results_per_time]
    return {'full_results': full_results, 'config': cfg, 'raw_df': data}


def run_single(
    data: pd.DataFrame,
    cfg: DictConfig,
    status: Optional[tqdm] = None,
    seed: Optional[int] = None,
) -> List[EstimateResult]:
    _, Y, W, T = build_arrays(data, cfg)
    treatment_path = np.asarray(cfg.model.treatment_path, dtype=int)
    control_path = np.asarray(cfg.model.control_path, dtype=int)

    t_max = min(T, treatment_path.shape[0])
    t_max = min(t_max, control_path.shape[0])

    n_units = W.shape[0]
    results = []
    mu_hats = np.zeros((t_max,), dtype=float)
    ipw_mat = np.zeros((n_units, t_max), dtype=float)
    Ap_mat = np.zeros((n_units, t_max), dtype=float)
    G1_mat = np.zeros((n_units, t_max), dtype=float)

    mu_hats_control = np.zeros((t_max,), dtype=float) if control_path is not None else None
    ipw_mat_control = np.zeros((n_units, t_max), dtype=float) if control_path is not None else None
    Ap_mat_control = np.zeros((n_units, t_max), dtype=float) if control_path is not None else None
    G1_mat_control = np.zeros((n_units, t_max), dtype=float) if control_path is not None else None
    for t_idx in range(t_max):
        if status is not None:
            status.set_description_str(f'Empirical t={t_idx + 1}/{t_max}')
            status.refresh()

        # ---- Treated path ----
        mask_treat = np.all(W[:, : t_idx + 1] == treatment_path[None, : t_idx + 1], axis=1)
        unit_idx_treat = np.where(mask_treat)[0]
        n_path_treat = unit_idx_treat.size
        y_treat = Y[unit_idx_treat, t_idx].astype(float)
        mu_hat_treat = np.mean(y_treat)
        pi_hat_treat = n_path_treat / n_units
        ipw_term_treat = np.full((n_units,), mu_hat_treat, dtype=float)
        ipw_term_treat[unit_idx_treat] += (y_treat - mu_hat_treat) / pi_hat_treat
        psi_treat = ipw_term_treat - mu_hat_treat

        mu_hats[t_idx] = mu_hat_treat
        ipw_mat[:, t_idx] = ipw_term_treat

        # ---- Optional control path ----
        if control_path is None:
            results.append(
                EstimateResult(
                    time=t_idx,
                    mu=mu_hat_treat,
                    residual=mu_hat_treat,
                    baseline=0.0,
                    trans_aug_mean=0.0,
                    n_path=n_path_treat,
                    n_total=n_units,
                    psi_values=psi_treat,
                )
            )
            continue

        mask_control = np.all(W[:, : t_idx + 1] == control_path[None, : t_idx + 1], axis=1)
        unit_idx_control = np.where(mask_control)[0]
        n_path_control = unit_idx_control.size
        y_control = Y[unit_idx_control, t_idx].astype(float)
        mu_hat_control = np.mean(y_control)
        pi_hat_control = n_path_control / n_units
        ipw_term_control = np.full((n_units,), mu_hat_control, dtype=float)
        ipw_term_control[unit_idx_control] += (y_control - mu_hat_control) / pi_hat_control
        psi_control = ipw_term_control - mu_hat_control

        mu_hats_control[t_idx] = mu_hat_control
        ipw_mat_control[:, t_idx] = ipw_term_control

        ate = mu_hat_treat - mu_hat_control
        results.append(
            EstimateResult(
                time=t_idx,
                mu=ate,
                mu_treatment=mu_hat_treat,
                mu_control=mu_hat_control,
                residual=ate,
                baseline=0.0,
                trans_aug_mean=0.0,
                n_path=min(n_path_treat, n_path_control),
                n_path_treatment=n_path_treat,
                n_path_control=n_path_control,
                n_total=n_units,
                psi_values=psi_treat - psi_control,
            )
        )

    ci_level = cfg.eval.ci_level
    alpha = 1.0 - ci_level
    if control_path is None:
        lower, upper = compute_ci(
            mu_hat=mu_hats,
            ipw_residual=ipw_mat,
            A_p=Ap_mat,
            G1=G1_mat,
            alpha=alpha,
            variance_type=cfg.eval.variance_type,
            band_type=cfg.eval.band_type,
            n_bootstrap=cfg.eval.n_boot,
            seed=cfg.exp.seed,
        )
    else:
        lower, upper = compute_ci(
            mu_hat=mu_hats,
            ipw_residual=ipw_mat,
            A_p=Ap_mat,
            G1=G1_mat,
            mu_hat_control=mu_hats_control,
            ipw_residual_control=ipw_mat_control,
            A_p_control=Ap_mat_control,
            G1_control=G1_mat_control,
            alpha=alpha,
            variance_type=cfg.eval.variance_type,
            band_type=cfg.eval.band_type,
            n_bootstrap=cfg.eval.n_boot,
            seed=cfg.exp.seed,
        )
    for res, lo, hi in zip(results, lower, upper):
        res['ci_lower'] = lo
        res['ci_upper'] = hi

    return results
