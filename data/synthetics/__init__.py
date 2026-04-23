from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numba import njit, prange
from omegaconf import DictConfig

EPS = 1e-12


@dataclass(frozen=True)
class ModelParams:
    x_to_x: float
    x_to_y: float
    w_to_x: float
    w_to_y: float
    y_to_y: float
    y_to_x: float
    eta: float
    mu_0: float
    mu_x: float
    mu_y: float
    sigma_0: float
    sigma_x: float
    sigma_y: float
    treatment_effect: np.ndarray  # (T, K)
    orthogonal_matrix: np.ndarray  # (d, d)


def load_data(
    cfg: DictConfig,
    model_params: Optional[ModelParams] = None,
    unit: str = 'id',
    time: str = 'date',
    feature: str = 'feature',
    value: str = 'value',
    covariates: Optional[Sequence[str]] = None,
    treatment: str = 'W',
    outcome: str = 'Y',
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    # Generate data
    X, Y, W, params = _generate_data(cfg, rng, model_params, seed=seed)

    # Covariate names
    if covariates is None:
        cov_names = [f'X{k + 1}' for k in range(cfg.d)]
    else:
        cov_names = list(covariates)
        if len(cov_names) != cfg.d:
            raise ValueError('Length of `covariates` must match `cfg.d`.')

    # Transform to long-format DataFrame
    rows = []
    ids = np.arange(1, cfg.n + 1, dtype=int)
    for i in range(cfg.n):
        for t in range(cfg.T):
            time_idx = t
            rows.append((ids[i], time_idx, outcome, Y[i, t]))
            rows.append((ids[i], time_idx, treatment, W[i, t]))
            for k, name in enumerate(cov_names):
                rows.append((ids[i], time_idx, name, X[i, t, k]))
    df = pd.DataFrame(rows, columns=[unit, time, feature, value])

    # Prepare sequential treatments
    all_treatment_paths = _extract_unique_treatment_paths(
        df=df,
        unit=unit,
        time=time,
        feature=feature,
        value=value,
        treatment=treatment,
    )

    # ground truth
    truth = {}
    mu_by_path = {}
    for path in all_treatment_paths:
        mu_by_path[path] = generate_ground_truth(params, path, cfg, seed=seed)
    truth['mu_by_path'] = mu_by_path

    if verbose:
        print(
            f'[synthetic datasets] df={df.shape}, n={cfg.n}, T={cfg.T}, '
            f'd={cfg.d}, K={cfg.K}, model={cfg.model_type}, assign={cfg.assign_mode}.'
        )

    return df, {
        'truth': truth,
        'covariate': ', '.join(cov_names),
        'outcome': outcome,
    }


def generate_params(cfg: DictConfig, seed: Optional[int] = None) -> ModelParams:
    x_to_x = cfg.x_to_x
    x_to_y = cfg.x_to_y
    w_to_x = cfg.w_to_x
    w_to_y = cfg.w_to_y
    y_to_y = cfg.y_to_y
    y_to_x = cfg.y_to_x
    eta = cfg.x_nonlinear
    t_idx = np.arange(1, cfg.T + 1, dtype=float)

    # Treatment effect over time: τ(t)
    w_base = np.asarray(cfg.w_base, dtype=float)
    w_amp = np.asarray(cfg.w_amp, dtype=float)
    # Range of curve is [0, 1]
    if cfg.treat_type == 'exp':
        curve = 1 - 3 * np.exp(-t_idx / 4.0) + 2 * np.exp(-3.0 * t_idx / 5.0)
    elif cfg.treat_type == 'seasonal':
        curve = (np.sin(0.7 * t_idx) + 1) / 2.0
    elif cfg.treat_type == 'static':
        curve = 0.1 * np.ones(cfg.T, dtype=float)
    else:
        raise ValueError('Invalid treat_type.')
    tau_time = w_base + w_amp * curve  # (T,)
    treatment_effect = np.zeros((cfg.T, cfg.K))
    treat_idx = 1 if cfg.K > 1 else 0
    treatment_effect[:, treat_idx] = tau_time

    # Orthogonal matrix sampled from Haar measure
    orthogonal_matrix = get_random_orthogonal_matrix(cfg.d, seed=seed)

    return ModelParams(
        x_to_x=x_to_x,
        x_to_y=x_to_y,
        w_to_x=w_to_x,
        w_to_y=w_to_y,
        y_to_y=y_to_y,
        y_to_x=y_to_x,
        eta=eta,
        mu_0=cfg.mu_0,
        mu_x=cfg.mu_x,
        mu_y=cfg.mu_y,
        sigma_0=cfg.sigma_0,
        sigma_x=cfg.sigma_x,
        sigma_y=cfg.sigma_y,
        treatment_effect=treatment_effect,
        orthogonal_matrix=orthogonal_matrix,
    )


def _generate_data(
    cfg: DictConfig,
    rng: np.random.Generator,
    params: Optional[ModelParams] = None,
    seed: Optional[int] = None,
    path: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ModelParams]:
    n, T, d, K = cfg.n, cfg.T, cfg.d, cfg.K

    # Setting params
    if params is None:
        params = generate_params(cfg=cfg, seed=seed)
    covariate_params = {
        'X': params.x_to_x,
        'Y': params.y_to_x,
        'W': params.w_to_x,
        'noise_mean': params.mu_x,
        'noise_std': params.sigma_x,
        'F': cfg.lorenz_force,
        'dt': cfg.timestep,
        'Q': params.orthogonal_matrix,
    }
    outcome_params = {
        'X': params.x_to_y,
        'Y': params.y_to_y,
        'W': params.w_to_y,
        'noise_mean': params.mu_y,
        'noise_std': params.sigma_y,
        'mu': cfg.gauss_mean,
    }

    # Generate treatment assignment path: (n, T)
    if path is None:
        W = _draw_W(cfg, rng)
    else:
        path_arr = np.asarray(path, dtype=int).reshape(-1)
        if path_arr.size != T:
            raise ValueError('len(path) must strictly match T.')
        if np.any((path_arr < 0) | (path_arr >= K)):
            raise ValueError('Each element of path must be an integer in the range 0..K-1.')
        W = np.repeat(path_arr[None, :], n, axis=0)

    # initial state
    X, Y = _generate_initial(
        n=n,
        T=T,
        d=d,
        W0=W[:, 0],
        treatment_effect=params.treatment_effect[0],
        mu=params.mu_0,
        sigma=params.sigma_0,
        outcome_params=outcome_params,
        reg_type=cfg.reg_type,
        rng=rng,
    )
    for t in range(1, T):
        X[:, t, :] = _generate_covariate(
            X[:, t - 1, :],
            Y[:, t - 1],
            W[:, t - 1],
            params.treatment_effect[t],
            params=covariate_params,
            trans_type=cfg.trans_type,
            rng=rng,
        )
        Y[:, t] = _generate_outcome(
            X[:, t, :],
            Y[:, t - 1],
            W[:, t],
            params.treatment_effect[t],
            params=outcome_params,
            reg_type=cfg.reg_type,
            rng=rng,
        )
    return X, Y, W, params


def generate_ground_truth(
    params: ModelParams,
    path: Sequence[int],
    cfg: DictConfig,
    N_mc: int = 1000000,
    chunk: int = 100000,
    seed: Optional[int] = None,
) -> np.ndarray:
    if len(path) != cfg.T:
        raise ValueError('len(path) must strictly match T.')
    if not all((isinstance(k, (int, np.integer)) and 0 <= k < cfg.K) for k in path):
        raise ValueError('Each element of path must be an integer in the range 0..K-1.')

    rng = np.random.default_rng(seed)
    muY = np.zeros(cfg.T, dtype=float)
    simulated = 0
    orig_n = cfg.n
    while simulated < N_mc:
        cfg.n = min(chunk, N_mc - simulated)
        _, Y, _, _ = _generate_data(cfg, rng, params=params, path=path)
        muY += Y.sum(axis=0)
        simulated += cfg.n
    cfg.n = orig_n

    return muY / simulated


def _generate_initial(
    n: int,
    T: int,
    d: int,
    W0: np.ndarray,
    treatment_effect: np.ndarray,
    mu: float,
    sigma: float,
    outcome_params: dict,
    reg_type: str = 'friedman',
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    X = np.zeros((n, T, d), float)  # (n, T, d)
    Y = np.zeros((n, T), float)  # (n, T)

    X[:, 0, :] = rng.normal(loc=mu, scale=sigma, size=(n, d))
    Y[:, 0] = _generate_outcome(
        X[:, 0, :],
        np.zeros(n),
        W0,
        treatment_effect,
        params=outcome_params,
        reg_type=reg_type,
        rng=rng,
    )

    return X, Y


@njit(inline='always')
def _compute_drift_single(x: np.ndarray, forcing: float, d: int, out: np.ndarray) -> None:
    for j in range(d):
        p1 = x[(j + 1) % d]
        m1 = x[(j - 1 + d) % d]
        m2 = x[(j - 2 + d) % d]

        out[j] = (p1 - m2) * m1 - x[j] + forcing


@njit(parallel=True, fastmath=True)
def _generate_lorenz_core(
    X_prev: np.ndarray,
    Y_prev: np.ndarray,
    W_prev: np.ndarray,
    dt: float,
    n_substeps: int,
    F: float,
    param_Y: float,
    param_W: float,
) -> np.ndarray:
    n, d = X_prev.shape
    actual_dt = dt / n_substeps

    X_next = np.empty((n, d), dtype=X_prev.dtype)
    for i in prange(n):
        x_curr = X_prev[i].copy()
        forcing = F + param_Y * Y_prev[i] + param_W * W_prev[i]

        k1 = np.empty(d)
        k2 = np.empty(d)
        k3 = np.empty(d)
        k4 = np.empty(d)
        temp_x = np.empty(d)

        # Time integration loop
        for _ in range(n_substeps):
            # k1
            _compute_drift_single(x_curr, forcing, d, k1)

            # k2
            for j in range(d):
                temp_x[j] = x_curr[j] + k1[j] * (actual_dt / 2.0)
            _compute_drift_single(temp_x, forcing, d, k2)

            # k3
            for j in range(d):
                temp_x[j] = x_curr[j] + k2[j] * (actual_dt / 2.0)
            _compute_drift_single(temp_x, forcing, d, k3)

            # k4
            for j in range(d):
                temp_x[j] = x_curr[j] + k3[j] * actual_dt
            _compute_drift_single(temp_x, forcing, d, k4)

            # Update state
            for j in range(d):
                x_curr[j] += (actual_dt / 6.0) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])

        X_next[i] = x_curr

    return X_next


def _generate_covariate(
    X_prev: np.ndarray,
    Y_prev: np.ndarray,
    W_prev: np.ndarray,
    treatment_effect: float,
    params: dict,
    trans_type: str,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    n, d = X_prev.shape

    if trans_type == 'lorenz':
        total_dt = params['dt']
        integration_dt = 0.01
        n_substeps = int(np.ceil(total_dt / integration_dt))
        X_curr = _generate_lorenz_core(
            X_prev,
            Y_prev,
            W_prev,
            total_dt,
            n_substeps,
            params['F'],
            params['Y'],
            params['W'],
        )

        sde_noise_scale = params['noise_std'] * np.sqrt(total_dt)
        noise = rng.normal(loc=0.0, scale=sde_noise_scale, size=(n, d))
        return X_curr + noise

    elif trans_type == 'rotated_lorenz':
        total_dt = params['dt']
        integration_dt = 0.01
        n_substeps = int(np.ceil(total_dt / integration_dt))

        Z_prev = X_prev @ params['Q'].T
        Z_curr = _generate_lorenz_core(
            Z_prev,
            Y_prev,
            W_prev,
            total_dt,
            n_substeps,
            params['F'],
            params['Y'],
            params['W'],
        )

        X_curr = Z_curr @ params['Q']
        sde_noise_scale = params['noise_std'] * np.sqrt(total_dt)
        noise = rng.normal(loc=0.0, scale=sde_noise_scale, size=(n, d))

        return X_curr + noise

    else:
        raise ValueError(f'Unsupported trans_type `{trans_type}`')


def _generate_outcome(
    X_t: np.ndarray,
    Y_prev: np.ndarray,
    W_t: np.ndarray,
    treatment_effect: np.ndarray,
    params: dict,
    reg_type: str = 'friedman',
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    n, K = len(X_t), len(treatment_effect)
    e = np.zeros((n, K))
    e[np.arange(n), W_t] = 1.0  # 0 if W_t == 0; else 1
    if rng is None:
        rng = np.random.default_rng()
    y_noise = rng.normal(loc=0.0, scale=params['noise_std'], size=n)

    # Generate baseline_mean
    if reg_type == 'friedman':
        baseline_mean = params['X'] * friedman_func(X_t, scale=0.1) + params['Y'] * Y_prev
    elif reg_type == 'tanh':
        baseline_mean = params['X'] * np.tanh(X_t.sum(axis=1)) + params['Y'] * Y_prev
    elif reg_type == 'gauss_sin':
        covariate_effect = np.exp(-((X_t[:, 0] - params['mu']) ** 2) / 5) + np.sin(X_t[:, 1] * X_t[:, 2] / 3)
        baseline_mean = params['X'] * covariate_effect + params['Y'] * np.log(1 + np.exp(Y_prev))
    else:
        raise ValueError(f'Unsupported reg_type `{reg_type}` for nonlinear model.')

    Y = baseline_mean + params['W'] * (treatment_effect @ e.T) + y_noise

    return Y


def _draw_W(
    cfg: DictConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    T, K = cfg.T, cfg.K
    if cfg.assign_mode == 'always':
        if cfg.hard_assign:
            base = np.full(K, cfg.n // K, dtype=int)
            base[: (cfg.n % K)] += 1
            cls = np.repeat(np.arange(K, dtype=int), base)
            rng.shuffle(cls)
            return np.repeat(cls[:, None], T, axis=1)
        else:
            cls = rng.integers(low=0, high=K, size=cfg.n)
            return np.repeat(cls[:, None], T, axis=1)
    elif cfg.assign_mode in ['random', 'markov']:
        raise NotImplementedError
    else:
        raise ValueError('Unknown assign_mode.')


def _extract_unique_treatment_paths(
    df: pd.DataFrame,
    unit: str = 'id',
    time: str = 'date',
    value: str = 'value',
    feature: str = 'feature',
    treatment: str = 'W',
) -> List[Tuple[int, ...]]:
    w_wide = df[df[feature] == treatment].pivot(index=unit, columns=time, values=value)
    if w_wide.shape[1] > 0:
        w_wide = w_wide.reindex(sorted(w_wide.columns), axis=1)
    if w_wide.isna().any().any():
        raise ValueError('treatment entries contain NaN. Dataframe must be complete before path extraction.')
    w_wide = w_wide.apply(pd.to_numeric, errors='raise').astype(int)
    unique_rows = w_wide.drop_duplicates()
    paths = [tuple(row) for row in unique_rows.to_numpy().tolist()]
    return sorted(paths)


def get_random_orthogonal_matrix(d: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    M = np.random.randn(d, d)
    Q, R = np.linalg.qr(M)
    diag_R = np.diagonal(R)
    ph = diag_R / np.abs(diag_R)
    Q = np.multiply(Q, ph, Q)

    return Q


def friedman_func(X: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if X.shape[1] < 5:
        raise ValueError('Input data must have at least 5 features.')

    term1 = 10 * np.sin(np.pi * X[:, 0] * X[:, 1])
    term2 = 20 * (X[:, 2] - 0.5) ** 2
    term3 = 10 * X[:, 3]
    term4 = 5 * X[:, 4]

    return (term1 + term2 + term3 + term4) * scale
