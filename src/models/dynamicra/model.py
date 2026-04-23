from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict, Union, cast

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.models.dynamicra.module.regmodel import RegressionModel, get_regression_factory
from src.models.dynamicra.module.transmodel import TransitionModel, get_transition_factory
from src.utils.inference import compute_ci


def path_mask(W: np.ndarray, treatment_path: np.ndarray, t_idx: int) -> np.ndarray:
    if t_idx < 0:
        return np.ones(W.shape[0], dtype=bool)
    if t_idx >= W.shape[1]:
        raise ValueError(f't_idx={t_idx} out of range (0..{W.shape[1] - 1}).')
    prefix = treatment_path[: t_idx + 1]
    mask = np.asarray(np.all(W[:, : t_idx + 1] == prefix[None, :], axis=1))
    if not np.any(mask):
        raise RuntimeError(f'Positivity violation: no units match treatment_path[:{t_idx + 1}] = {prefix.tolist()}.')
    return mask


def _history_window_drop_cols(history_window: Optional[int], t_idx: int, x_dim: int) -> int:
    if history_window is None or history_window < 0:
        return 0
    if t_idx + 1 <= history_window:
        return 0
    old_time = t_idx - history_window
    return x_dim if old_time == 0 else x_dim + 1


@dataclass
class FoldManager:
    n_folds: int
    seed: Optional[int]
    n_samples: int
    splits: List[Tuple[np.ndarray, np.ndarray]]
    fold_assign: np.ndarray

    @classmethod
    def build(cls, n_samples: int, n_folds: int, seed: Optional[int]) -> 'FoldManager':
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = []
        fold_assign = -np.ones(n_samples, dtype=int)
        for fold_id, (tr, va) in enumerate(kf.split(np.arange(n_samples))):
            splits.append((tr, va))
            fold_assign[va] = fold_id
        if np.any(fold_assign < 0):
            raise RuntimeError('Fold assignment incomplete.')
        return cls(
            n_folds=n_folds,
            seed=seed,
            n_samples=n_samples,
            splits=splits,
            fold_assign=fold_assign,
        )


class EstimateResult(TypedDict, total=False):
    time: int
    mu: float
    mu_treatment: float
    mu_control: float
    residual: float
    baseline: float
    trans_aug_mean: float
    n_path: int
    n_path_treatment: int
    n_path_control: int
    n_total: int
    psi_values: np.ndarray
    ci_lower: float
    ci_upper: float


@dataclass
class _PathEstimateArtifacts:
    time_ls: List[int]
    results: List[EstimateResult]
    mu_hat: np.ndarray
    ipw_residual: np.ndarray
    G1: np.ndarray
    A_p: np.ndarray


class VectorizedFeatureBuilder:
    def __init__(self, X: np.ndarray, Y: np.ndarray, history_window: Optional[int] = None):
        self.X = np.asarray(X, dtype=float)
        if Y.ndim == 1:
            self.Y = Y.reshape(-1, 1).astype(float)
        elif Y.ndim == 2:
            self.Y = Y.reshape(Y.shape[0], Y.shape[1], 1).astype(float)
        else:
            self.Y = np.asarray(Y, dtype=float)

        self.N, self.T, self.D = self.X.shape
        self.history_window = history_window

        self._cache: Dict[int, np.ndarray] = {}

    def get_features(self, t_idx: int, indices: Optional[np.ndarray] = None) -> np.ndarray:
        if not (0 <= t_idx < self.T):
            raise ValueError(f't_idx={t_idx} is out of bounds (0..{self.T - 1})')

        if t_idx not in self._cache:
            self._materialize_cache_up_to(t_idx)

        features = self._cache[t_idx]

        if indices is not None:
            return cast(np.ndarray, features[indices])

        return features

    def _materialize_cache_up_to(self, target_t: int) -> None:
        start_t = 0
        for t in range(target_t, -1, -1):
            if t in self._cache:
                start_t = t + 1
                break

        for t in range(start_t, target_t + 1):
            if t == 0:
                new_features = self.X[:, 0, :].copy()
            else:
                prev_features = self._cache[t - 1]
                x_inc = self.X[:, t, :]
                y_inc = self.Y[:, t - 1, :]
                new_features = np.concatenate([prev_features, x_inc, y_inc], axis=1)

            drop_cols = _history_window_drop_cols(self.history_window, t, self.D)
            if drop_cols > 0:
                new_features = new_features[:, drop_cols:]
            self._cache[t] = new_features


class DynamicRAEstimator:
    def __init__(
        self,
        method: str = 'dynamicra',
        regmodel_name: str = 'ols',
        transmodel_name: str = 'var',
        n_folds: int = 5,
        n_forward_mc: int = 128,
        history_window: Optional[int] = -1,
        seed: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.method = method
        self.regmodel_name = regmodel_name
        self.transmodel_name = transmodel_name
        self.n_folds = n_folds
        self.n_forward_mc = n_forward_mc
        self.history_window = history_window
        self.seed = seed
        self.verbose = verbose
        self._x_dim: Optional[int] = None

    def _trim_history_window(self, particles: np.ndarray, next_time_idx: int) -> np.ndarray:
        if self.history_window is None:
            return particles
        if self._x_dim is None:
            raise RuntimeError('x_dim is not initialized before transition.')
        drop_cols = _history_window_drop_cols(self.history_window, next_time_idx, self._x_dim)
        if drop_cols <= 0:
            return particles
        return particles[:, :, drop_cols:]

    def _batch_predict(
        self,
        models: List[RegressionModel],
        fold_assign: np.ndarray,
        particles: np.ndarray,  # Shape (N, S, D)
    ) -> np.ndarray:
        N, S, D = particles.shape
        results = np.zeros((N, S), dtype=float)

        for fold_id, model in enumerate(models):
            idx = np.where(fold_assign == fold_id)[0]
            if idx.size == 0:
                continue

            batch_X = particles[idx].reshape(-1, D)

            batch_X = np.nan_to_num(batch_X, nan=0.0, posinf=1e10, neginf=-1e10)
            batch_X = np.clip(batch_X, -1e10, 1e10)

            preds = model.predict(batch_X)

            results[idx] = preds.reshape(len(idx), S)

        return results

    def _batch_transition(
        self,
        trans_models_tau: List[TransitionModel],
        fold_assign: np.ndarray,
        particles: np.ndarray,  # (N, S, D)
        tau_idx: int,
    ) -> np.ndarray:
        N, S, D = particles.shape
        if self._x_dim is None:
            raise RuntimeError('x_dim is not initialized before transition.')
        d_inc = self._x_dim + 1
        total_dim = D + d_inc
        final_particles = np.zeros((N, S, total_dim), dtype=float)

        for fold_id, model in enumerate(trans_models_tau):
            idx = np.where(fold_assign == fold_id)[0]
            if idx.size == 0:
                continue

            n_sub = len(idx)
            subset_P = particles[idx]  # (n_sub, S, D)
            flat_P = subset_P.reshape(-1, D)  # (n_sub * S, D)
            seed = self._transition_seed(tau_idx=tau_idx, fold=fold_id)
            rng = np.random.default_rng(seed)

            z_input = rng.standard_normal((n_sub * S, 1, d_inc))
            next_cols = model.sample_next(flat_P, n_samples=1, rng=rng, z=z_input).reshape(n_sub, S, -1)
            updated_subset = np.concatenate([subset_P, next_cols], axis=2)
            final_particles[idx] = updated_subset

        final_particles = np.nan_to_num(final_particles, nan=0.0, posinf=1e10, neginf=-1e10)
        final_particles = np.clip(final_particles, -1e10, 1e10)

        return self._trim_history_window(final_particles, next_time_idx=tau_idx + 1)

    def _transition_seed(self, tau_idx: int, fold: int) -> int:
        seq = np.random.SeedSequence(self.seed, spawn_key=(tau_idx, fold))
        return int(seq.generate_state(1, dtype=np.uint32)[0])

    @staticmethod
    def _normalize_times(times: Optional[Union[int, List[int]]], t_horizon: int) -> List[int]:
        if t_horizon <= 0:
            raise ValueError('t_horizon must be positive.')
        if times is None:
            time_ls = list(range(t_horizon))
        elif isinstance(times, int):
            time_ls = [times]
        elif isinstance(times, list):
            time_ls = times
        else:
            raise ValueError('times must be an int, list of int, or None.')
        time_ls = sorted(set(time_ls))
        if any((t < 0) or (t >= t_horizon) for t in time_ls):
            raise ValueError(f'times must lie in [0, {t_horizon - 1}]')
        return time_ls

    def _estimate_single_path(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        W: np.ndarray,
        time_ls: List[int],
        path: np.ndarray,
        status_bar: Optional[tqdm] = None,
        status_prefix: str = '',
    ) -> _PathEstimateArtifacts:
        if path is None:
            raise ValueError('path must be provided explicitly.')
        path = np.asarray(path, dtype=int)
        n, T, self._x_dim = X.shape
        t_horizon = min(T, path.shape[0])
        if any((t < 0) or (t >= t_horizon) for t in time_ls):
            raise ValueError(f'times must lie in [0, {t_horizon - 1}]')

        time_to_col = {t_idx: j for j, t_idx in enumerate(time_ls)}
        m = len(time_ls)
        t_max = time_ls[-1]

        # Cross-fitting setup
        fold_manager = FoldManager.build(n_samples=n, n_folds=self.n_folds, seed=self.seed)
        fold_assign = fold_manager.fold_assign.copy()

        # Storage for models
        reg_by_t: Dict[int, List[RegressionModel]] = {}
        trans_by_tau: List[List[TransitionModel]] = []
        mask_by_tau: Dict[int, np.ndarray] = {}
        pi_by_tau: Dict[int, float] = {}

        # --- Vectorized Storage ---
        S = self.n_forward_mc
        feature_builder = VectorizedFeatureBuilder(X, Y, history_window=self.history_window)

        phi0_all = feature_builder.get_features(0)  # (N, D)
        P_base = np.tile(phi0_all[:, None, :], (1, S, 1))  # (N, S, D)

        R_buffers: Dict[int, np.ndarray] = {}
        C_buffers: Dict[int, np.ndarray] = {}

        # Results storage
        mu_hats = np.zeros((m,), dtype=float)
        ipw_mat = np.zeros((n, m), dtype=float)
        G1_mat = np.zeros((n, m), dtype=float)
        Ap_mat = np.zeros((n, m), dtype=float)
        results: List[EstimateResult] = []

        for t_idx in range(t_max + 1):
            # print(f'T={t_idx}/{t_max}')
            if status_bar is not None:
                prefix = f'{status_prefix} ' if status_prefix else ''
                status_bar.set_description_str(f'{prefix}t={t_idx + 1}/{t_max + 1}')
                status_bar.refresh()

            phi_real_all = feature_builder.get_features(t_idx)

            # 1) Transition Learning & New Particle Generation
            if t_idx >= 1 and self.method in ['dynamicra', 'gformula']:
                trans_by_fold = []

                C_all_tau = feature_builder.get_features(t_idx - 1)
                X_next_all = X[:, t_idx, :]
                Y_curr_all = Y[:, t_idx - 1].reshape(-1, 1)
                S_all_tau = np.concatenate([X_next_all, Y_curr_all], axis=1).astype(float)

                mask = path_mask(W, path, t_idx - 1)
                mask_by_tau[t_idx - 1] = mask
                pi_by_tau[t_idx - 1] = np.sum(mask) / n

                for fold_id, (train_idx, _) in enumerate(fold_manager.splits):
                    # extract only units on the path
                    unit_idx = train_idx[mask[train_idx]]

                    trans_model = get_transition_factory(
                        model_name=self.transmodel_name,
                        hidden_dim=4 * C_all_tau[unit_idx].shape[1],
                        seed=self._transition_seed(tau_idx=t_idx - 1, fold=fold_id),
                    )
                    trans_model = trans_model.fit(C_all_tau[unit_idx], S_all_tau[unit_idx])
                    trans_by_fold.append(trans_model)
                trans_by_tau.append(trans_by_fold)

                if self.method == 'dynamicra':
                    R_buffers[t_idx - 1] = np.tile(phi_real_all[:, None, :], (1, S, 1))
                    C_base = np.tile(C_all_tau[:, None, :], (1, S, 1))
                    C_buffers[t_idx - 1] = self._batch_transition(trans_by_fold, fold_assign, C_base, t_idx - 1)

            # 2) Regression Learning
            if t_idx in time_ls:
                reg_by_fold = []
                for fold_id, (train_idx, _) in enumerate(fold_manager.splits):
                    mask = path_mask(W[train_idx], path, t_idx)
                    idx = train_idx[mask]
                    reg_model = get_regression_factory(self.regmodel_name, seed=self.seed)
                    reg = reg_model.fit(phi_real_all[idx], Y[idx, t_idx])
                    reg_by_fold.append(reg)
                reg_by_t[t_idx] = reg_by_fold

            if t_idx >= 1 and self.method in ['dynamicra', 'gformula']:
                tau_prev = t_idx - 1
                models = trans_by_tau[tau_prev]

                update_blocks = []
                update_keys = []
                for tau_old in list(R_buffers.keys()):
                    if tau_old <= t_idx - 2:
                        update_blocks.append(R_buffers[tau_old])
                        update_keys.append(('R', tau_old))
                        update_blocks.append(C_buffers[tau_old])
                        update_keys.append(('C', tau_old))

                if not update_blocks:
                    P_base = self._batch_transition(models, fold_assign, P_base, tau_prev)
                else:
                    blocks = [P_base] + update_blocks
                    stacked_particles = np.concatenate(blocks, axis=0)
                    stacked_folds = np.tile(fold_assign, len(blocks))
                    updated = self._batch_transition(
                        models,
                        stacked_folds,
                        stacked_particles,
                        tau_prev,
                    )

                    n_block = P_base.shape[0]
                    P_base = updated[:n_block]
                    offset = n_block
                    for kind, tau_old in update_keys:
                        block = updated[offset : offset + n_block]
                        if kind == 'R':
                            R_buffers[tau_old] = block
                        else:
                            C_buffers[tau_old] = block
                        offset += n_block

            # 4) Estimation
            if t_idx in time_ls:
                col = time_to_col[t_idx]
                models_reg = reg_by_t[t_idx]

                mask_t = path_mask(W, path, t_idx)
                idx_t = np.where(mask_t)[0]
                pi_t = idx_t.size / n
                residual_vals = np.empty((idx_t.shape[0],), float)
                if idx_t.size > 0:
                    for fold_id, model in enumerate(models_reg):
                        is_target = (fold_assign == fold_id) & mask_t
                        target_indices = np.where(is_target)[0]
                        if target_indices.size == 0:
                            continue

                        # predict input: (N_sub, D)
                        X_target = phi_real_all[target_indices]
                        y_pred_fold = model.predict(X_target)

                        rel_pos = np.searchsorted(idx_t, target_indices)
                        residual_vals[rel_pos] = Y[target_indices, t_idx] - y_pred_fold

                res_mean = residual_vals.mean() if residual_vals.size > 0 else 0.0
                if self.method in ['aipw']:
                    G1_col = np.zeros(n)
                    base_mean = 0.0
                else:
                    G1_preds_S = self._batch_predict(models_reg, fold_assign, P_base)
                    G1_col = np.mean(G1_preds_S, axis=1)
                    base_mean = np.mean(G1_col)

                Ap_col = np.zeros(n, dtype=float)
                if self.method == 'dynamicra':
                    for tau_idx in range(t_idx):
                        if (tau_idx not in R_buffers) or (pi_by_tau[tau_idx] <= 1e-12):
                            continue

                        # (N, S)
                        preds_R = self._batch_predict(models_reg, fold_assign, R_buffers[tau_idx])
                        preds_C = self._batch_predict(models_reg, fold_assign, C_buffers[tau_idx])

                        diff_mean = np.mean(preds_R - preds_C, axis=1)
                        mask_tau = mask_by_tau[tau_idx]

                        term = np.zeros(n)
                        term[mask_tau] = diff_mean[mask_tau] / pi_by_tau[tau_idx]

                        Ap_col += term

                aug_mean = np.mean(Ap_col)

                # --- Final Logic ---
                ipw_term = np.zeros((n,), float)
                ipw_term[idx_t] = residual_vals / pi_t
                if self.method == 'gformula':
                    mu_hat = base_mean
                    psi = np.zeros(n, dtype=float)
                elif self.method == 'aipw':
                    if t_idx == 0:
                        idx_prev = np.arange(n)
                        is_prev = np.ones(n, dtype=bool)
                    else:
                        mask_prev = path_mask(W, path, t_idx - 1)
                        idx_prev = np.where(mask_prev)[0]
                        is_prev = mask_prev

                    m_preds_prev = np.zeros(n, dtype=float)
                    for fold_id, model in enumerate(models_reg):
                        is_target = (fold_assign == fold_id) & is_prev
                        target_indices = np.where(is_target)[0]
                        if target_indices.size > 0:
                            batch_X = phi_real_all[target_indices]
                            batch_X = np.nan_to_num(batch_X, nan=0.0, posinf=1e10, neginf=-1e10)
                            batch_X = np.clip(batch_X, -1e10, 1e10)
                            m_preds_prev[target_indices] = model.predict(batch_X)

                    m_mean_prev = m_preds_prev[idx_prev].mean() if idx_prev.size > 0 else 0.0
                    mu_hat = res_mean + m_mean_prev
                    psi = np.zeros(n, dtype=float)
                elif self.method == 'dynamicra':
                    mu_hat = base_mean + res_mean + aug_mean
                    psi = ipw_term + Ap_col + G1_col - mu_hat
                else:
                    raise ValueError(f'Unknown method: {self.method}')

                mu_hats[col] = mu_hat
                ipw_mat[:, col] = ipw_term
                G1_mat[:, col] = G1_col
                Ap_mat[:, col] = Ap_col
                results.append(
                    EstimateResult(
                        time=t_idx,
                        mu=mu_hat,
                        residual=res_mean,
                        baseline=base_mean,
                        trans_aug_mean=aug_mean,
                        n_path=idx_t.size,
                        n_total=n,
                        psi_values=psi,
                    )
                )

        return _PathEstimateArtifacts(
            time_ls=time_ls,
            results=results,
            mu_hat=mu_hats,
            ipw_residual=ipw_mat,
            G1=G1_mat,
            A_p=Ap_mat,
        )

    def estimate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        W: np.ndarray,
        times: Optional[Union[int, List[int]]],
        treatment_path: np.ndarray,
        alpha: float = 0.05,
        variance_type: str = 'moment',
        band_type: str = 'pointwise',
        n_bootstrap: int = 1000,
        control_path: Optional[np.ndarray] = None,
        status_bar: Optional[tqdm] = None,
    ) -> List[EstimateResult]:
        if treatment_path is None:
            raise ValueError('treatment_path must be provided explicitly.')
        treatment_path = np.asarray(treatment_path, dtype=int)
        _, T, _ = X.shape
        t_horizon = min(T, treatment_path.shape[0])

        control_arr: Optional[np.ndarray]
        if control_path is None:
            control_arr = None
        else:
            control_arr = np.asarray(control_path, dtype=int)
            t_horizon = min(t_horizon, control_arr.shape[0])

        time_ls = self._normalize_times(times, t_horizon=t_horizon)

        treated = self._estimate_single_path(
            X=X,
            Y=Y,
            W=W,
            time_ls=time_ls,
            path=treatment_path,
            status_bar=status_bar,
            status_prefix='DynamicRA[T]',
        )

        if control_arr is None:
            lower, upper = compute_ci(
                mu_hat=treated.mu_hat,
                ipw_residual=treated.ipw_residual,
                A_p=treated.A_p,
                G1=treated.G1,
                alpha=alpha,
                variance_type=variance_type,
                band_type=band_type,
                n_bootstrap=n_bootstrap,
                seed=self.seed,
            )
            for res, lo, hi in zip(treated.results, lower, upper):
                res['ci_lower'] = lo
                res['ci_upper'] = hi
            return treated.results

        control = self._estimate_single_path(
            X=X,
            Y=Y,
            W=W,
            time_ls=time_ls,
            path=control_arr,
            status_bar=status_bar,
            status_prefix='DynamicRA[C]',
        )

        lower, upper = compute_ci(
            mu_hat=treated.mu_hat,
            ipw_residual=treated.ipw_residual,
            A_p=treated.A_p,
            G1=treated.G1,
            mu_hat_control=control.mu_hat,
            ipw_residual_control=control.ipw_residual,
            A_p_control=control.A_p,
            G1_control=control.G1,
            alpha=alpha,
            variance_type=variance_type,
            band_type=band_type,
            n_bootstrap=n_bootstrap,
            seed=self.seed,
        )

        ate_results: List[EstimateResult] = []
        for treated_res, control_res, lo, hi in zip(treated.results, control.results, lower, upper):
            time_idx = treated_res['time']
            if time_idx != control_res['time']:
                raise RuntimeError('Treated/control results have mismatched time indices.')

            n_path_t = treated_res['n_path']
            n_path_c = control_res['n_path']
            n_total = treated_res['n_total']
            psi_t = np.asarray(treated_res['psi_values'], dtype=float)
            psi_c = np.asarray(control_res['psi_values'], dtype=float)

            ate_results.append(
                EstimateResult(
                    time=time_idx,
                    mu=treated_res['mu'] - control_res['mu'],
                    mu_treatment=treated_res['mu'],
                    mu_control=control_res['mu'],
                    residual=treated_res['residual'] - control_res['residual'],
                    baseline=treated_res['baseline'] - control_res['baseline'],
                    trans_aug_mean=treated_res['trans_aug_mean'] - control_res['trans_aug_mean'],
                    n_path=min(n_path_t, n_path_c),
                    n_path_treatment=n_path_t,
                    n_path_control=n_path_c,
                    n_total=n_total,
                    psi_values=psi_t - psi_c,
                    ci_lower=lo,
                    ci_upper=hi,
                )
            )

        return ate_results
