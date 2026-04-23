from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm


def compute_ci(
    mu_hat: np.ndarray,
    ipw_residual: np.ndarray,
    A_p: np.ndarray,
    G1: np.ndarray,
    mu_hat_control: Optional[np.ndarray] = None,
    ipw_residual_control: Optional[np.ndarray] = None,
    A_p_control: Optional[np.ndarray] = None,
    G1_control: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    variance_type: str = 'moment',
    band_type: str = 'pointwise',
    n_bootstrap: int = 500,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n, T = ipw_residual.shape
    if (
        mu_hat_control is not None
        and ipw_residual_control is not None
        and A_p_control is not None
        and G1_control is not None
    ):
        psi_control = ipw_residual_control + A_p_control + G1_control - mu_hat_control[np.newaxis, :]
    else:
        psi_control = np.zeros((n, T), dtype=float)
        mu_hat_control = np.zeros((T,), dtype=float)

    # ---- Construct influence function ----
    # IF: 1{path}/π*(Y-m) + A_p + G^{(1)} − μ̂
    psi_treated = ipw_residual + A_p + G1 - mu_hat[np.newaxis, :]  # (n, T)
    psi = psi_treated - psi_control
    mu = mu_hat - mu_hat_control

    sqrt_n = np.sqrt(n)
    omega = (psi**2).mean(axis=0)  # (T,)
    se = np.sqrt(np.maximum(omega, 0.0) / n)  # (T,)
    psi_centered = psi - psi.mean(axis=0, keepdims=True)

    if band_type not in ['pointwise', 'uniform']:
        raise ValueError('`band_type` must be either "pointwise" or "uniform".')

    if band_type == 'pointwise':
        if variance_type == 'moment':
            q_lo = norm.ppf(alpha / 2.0)
            q_hi = norm.ppf(1.0 - alpha / 2.0)

            lower = mu + q_lo * se
            upper = mu + q_hi * se
            return lower, upper
        elif variance_type == 'multiplier':
            rng = np.random.default_rng(seed)
            weights = rng.standard_normal((n_bootstrap, n))  # (B, n)
            replicates = weights @ psi_centered  # (B, T)
            replicates /= sqrt_n  # Approximate sqrt(n) * (mu_hat - mu)

            q_lo = np.quantile(replicates, alpha / 2.0, axis=0)
            q_hi = np.quantile(replicates, 1.0 - alpha / 2.0, axis=0)

            lower = mu - q_hi / sqrt_n
            upper = mu - q_lo / sqrt_n
            return lower, upper
        else:
            raise NotImplementedError(f'Invalid variance type was specified: {variance_type}.')
    else:
        if variance_type == 'multiplier':
            rng = np.random.default_rng(seed)
            weights = rng.standard_normal((n_bootstrap, n))  # (B, n)
            weights -= weights.mean(axis=1, keepdims=True)
            replicates = (weights @ psi_centered) / sqrt_n  # (B, T)

            max_t = np.max(np.abs(replicates), axis=1)
            c_alpha = np.quantile(max_t, 1.0 - alpha)

            lower = mu - c_alpha / sqrt_n
            upper = mu + c_alpha / sqrt_n
            return lower, upper
        else:
            raise NotImplementedError(f'Invalid variance type was specified: {variance_type}.')
