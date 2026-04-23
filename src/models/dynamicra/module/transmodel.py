from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR

EPS = 1e-12
MIN_POSDEF_EIG = 1e-8


def _ensure_pos_def(matrix: np.ndarray, min_eig: float = MIN_POSDEF_EIG) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, min_eig, None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


class TransitionModel:
    def fit(self, X_ctx: np.ndarray, Y_next: np.ndarray) -> 'TransitionModel':
        raise NotImplementedError

    def sample_next(
        self,
        X_ctx: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        z: Optional[np.ndarray] = None,
    ) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def get_transition_factory(model_name: str, hidden_dim: int = 10, seed: Optional[int] = None) -> TransitionModel:
    if model_name == 'mlp':
        return MLPNonlinearGaussianTransition(hidden_dim=hidden_dim, seed=seed, verbose=False)
    elif model_name == 'mlp-ziln':
        return MLPNonlinearZILNTransition(hidden_dim=hidden_dim, seed=seed, verbose=False)
    elif model_name == 'var':
        return VARTransition()
    else:
        raise ValueError(f'Unknown transition model name: {model_name}')


class MLPNonlinearGaussianTransition(TransitionModel):
    def __init__(
        self,
        hidden_dim: int = 16,
        n_layers: int = 3,
        lr: float = 5e-3,
        weight_decay: float = 1e-2,
        epochs: int = 2000,
        batch_size: int = 512,
        seed: Optional[int] = None,
        early_stopping: bool = True,
        patience: int = 50,
        min_delta_fraction: float = 1e-4,
        val_fraction: float = 0.1,
        grad_clip_norm: Optional[float] = 1.0,
        grad_clip_value: Optional[float] = None,
        use_residual_baseline: bool = False,
        verbose: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.early_stopping = early_stopping
        self.patience = max(1, patience)
        self.min_delta_fraction = max(0.0, min_delta_fraction)
        self._current_min_delta: Optional[float] = None
        self.val_fraction = val_fraction
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value
        self.use_residual_baseline = use_residual_baseline
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # linear baselines
        self._lin_coef_x = None
        self._lin_intercept_x = None
        self._lin_coef_y = None
        self._lin_intercept_y = None

        self._fitted: bool = False
        self._var_eps = 1e-4
        self._temp_x: float = 1.0
        self._temp_y: float = 1.0
        if seed is not None:
            torch.manual_seed(seed)

    @staticmethod
    def _mlp(d_in: int, d_out: int, hidden: int, n_layers: int, dropout: float = 0.3) -> nn.Sequential:
        layers: list[nn.Module] = []
        if n_layers == 1:
            layers.append(nn.Linear(d_in, d_out))
        else:
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.SiLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))

            layers.append(nn.Linear(hidden, d_out))

        return nn.Sequential(*layers)

    @staticmethod
    def _init_mlp_weights(net: nn.Module) -> None:
        for module in net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.01, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _fit_scaler(data: np.ndarray, tr_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if data.shape[1] == 0:
            empty = np.empty((0,), dtype=float)
            return data.copy(), empty, empty
        scaler = StandardScaler()
        scaler.fit(data[tr_idx])
        data_std = scaler.transform(data)
        return data_std, scaler.mean_, scaler.scale_

    @staticmethod
    def _gaussian_nll(pred: torch.Tensor, target: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        resid = target - pred
        if var.ndim == 1:
            log_var = torch.log(var)
            inv_var = torch.reciprocal(var)
            nll = 0.5 * (log_var.unsqueeze(0) + resid.pow(2) * inv_var.unsqueeze(0))
            return nll.sum(dim=-1).mean()
        if var.ndim == 2:
            if var.numel() == 0:
                return torch.tensor(0.0, device=target.device)
            if var.shape == resid.shape:
                log_var = torch.log(var)
                inv_var = torch.reciprocal(var)
                nll = 0.5 * (log_var + resid.pow(2) * inv_var)
                return nll.sum(dim=-1).mean()
            if var.shape[0] != var.shape[1]:
                raise ValueError('var must be 1D, 2D diag, or 2D Cholesky.')
            sol = torch.linalg.solve_triangular(var, resid.T, upper=False)
            quad = sol.pow(2).sum(dim=0)
            logdet = 2.0 * torch.log(torch.diagonal(var)).sum()
            nll = 0.5 * (logdet + quad)
            return nll.mean()
        raise ValueError('var must be 1D, 2D diag, or 2D Cholesky.')

    def fit(self, X_ctx: np.ndarray, Y_next: np.ndarray) -> 'MLPNonlinearGaussianTransition':
        N, d_in = X_ctx.shape
        d_next = Y_next.shape[1]
        d_x = d_next - 1
        Xnext = Y_next[:, :d_x]
        y = Y_next[:, -1:]

        # reset temperature scaling
        self._temp_x = 1.0
        self._temp_y = 1.0
        adaptive_min_delta: Optional[float] = None
        self._current_min_delta = None

        # split train/val
        if self.early_stopping and N > 1 and self.val_fraction > 0.0:
            v = min(max(1, round(self.val_fraction * N)), N - 1)
            perm = self.rng.permutation(N)
            tr_idx = perm[: N - v]
            va_idx = perm[N - v :]
        else:
            tr_idx = np.arange(N, dtype=np.int64)
            va_idx = None
        if tr_idx.size == 0:
            tr_idx = np.arange(N, dtype=np.int64)
            va_idx = None

        # normalization (fit on train only)
        X_std, self._x_mean, self._x_scale = self._fit_scaler(X_ctx, tr_idx)
        Xn_std, self._xnext_mean, self._xnext_scale = self._fit_scaler(Xnext, tr_idx)
        y_std, self._y_mean, self._y_scale = self._fit_scaler(y, tr_idx)

        # linear residual baselines
        self._fit_linear_baselines(X_std, Xn_std, y_std, tr_idx)

        # tensors
        X_t = torch.from_numpy(X_std.astype(np.float32)).to(self.device)
        Xn_t = torch.from_numpy(Xn_std.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y_std.astype(np.float32)).to(self.device)
        Xy_full = torch.cat([X_t, y_t], dim=1)

        self._net_y = self._mlp(d_in, 1, self.hidden_dim, self.n_layers).to(self.device)
        self._net_y_var = self._mlp(d_in, 1, self.hidden_dim, self.n_layers).to(self.device)
        self._init_mlp_weights(self._net_y)
        self._init_mlp_weights(self._net_y_var)

        in_x = d_in + 1
        self._net_x = self._mlp(in_x, d_x, self.hidden_dim, self.n_layers).to(self.device)
        self._net_x_var = self._mlp(in_x, d_x, self.hidden_dim, self.n_layers).to(self.device)
        self._init_mlp_weights(self._net_x)
        self._init_mlp_weights(self._net_x_var)

        # optimizer: weight decay only on weights, not biases/vars
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for _, p in self._net_x.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        for _, p in self._net_y.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        for _, p in self._net_x_var.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        for _, p in self._net_y_var.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = []
        if decay:
            groups.append({'params': decay, 'weight_decay': self.weight_decay})
        if no_decay:
            groups.append({'params': no_decay, 'weight_decay': 0.0})
        optimizer = optim.AdamW(groups, lr=self.lr, eps=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

        # validation tensors
        use_val = self.early_stopping and va_idx is not None and va_idx.size > 0
        if use_val:
            vi = torch.as_tensor(va_idx, dtype=torch.long, device=self.device)
            X_va, Xn_va, y_va = X_t.index_select(0, vi), Xn_t.index_select(0, vi), y_t.index_select(0, vi)
            Xy_va = Xy_full.index_select(0, vi)
            with torch.inference_mode():
                base_yv_val = self._baseline(X_va, self._lin_coef_y_t, self._lin_intercept_y_t, y_va)
                base_xv_val = self._baseline(Xy_va, self._lin_coef_x_t, self._lin_intercept_x_t, Xn_va)

        # training loop
        best = np.inf
        wait = 0
        best_state: Optional[Mapping[str, Any]] = None
        params_to_clip = []
        params_to_clip += list(self._net_x.parameters())
        params_to_clip += list(self._net_y.parameters())
        params_to_clip += list(self._net_x_var.parameters())
        params_to_clip += list(self._net_y_var.parameters())

        # training
        warmup = 10
        for epoch in range(self.epochs):
            if tr_idx.size > 0:
                self.rng.shuffle(tr_idx)
            seen = 0
            epoch_loss = 0.0
            for start in range(0, tr_idx.size, self.batch_size):
                sl = tr_idx[start : start + self.batch_size]
                if sl.size == 0:
                    continue
                si = torch.as_tensor(sl, dtype=torch.long, device=self.device)
                xb, xnb, yb = X_t.index_select(0, si), Xn_t.index_select(0, si), y_t.index_select(0, si)
                Xyb = Xy_full.index_select(0, si)

                # forward + backward
                optimizer.zero_grad(set_to_none=True)

                # stage 1: X -> y (residual target)
                base_y = self._baseline(xb, self._lin_coef_y_t, self._lin_intercept_y_t, yb)
                pred_y = self._net_y(xb)
                var_y = self._var_y(xb)
                nll_y = self._gaussian_nll(pred_y, yb - base_y, var_y)

                # stage 2: [X, y] -> Xnext (teacher forcing)
                base_x = self._baseline(Xyb, self._lin_coef_x_t, self._lin_intercept_x_t, xnb)
                pred_x = self._net_x(Xyb)
                var_x = self._var_x(Xyb)
                nll_x = self._gaussian_nll(pred_x, xnb - base_x, var_x)

                # combine losses
                loss = nll_y + (nll_x / d_x)
                loss.backward()
                if self.grad_clip_norm is not None:
                    mx = self.grad_clip_norm if epoch >= warmup else max(self.grad_clip_norm, 10.0)
                    torch.nn.utils.clip_grad_norm_(params_to_clip, mx)
                if self.grad_clip_value is not None:
                    torch.nn.utils.clip_grad_value_(params_to_clip, self.grad_clip_value)
                optimizer.step()

                bsz = xb.shape[0]
                seen += bsz
                epoch_loss += loss.item() * bsz

            epoch_loss = epoch_loss / max(1, seen)

            # validation
            if use_val:
                self._net_x.eval()
                self._net_y.eval()
                self._net_x_var.eval()
                self._net_y_var.eval()
                with torch.inference_mode():
                    var_x = self._var_x(Xy_va)
                    var_y = self._var_y(X_va)
                    pvy = self._net_y(X_va)
                    lvy = self._gaussian_nll(pvy, y_va - base_yv_val, var_y)
                    pvx = self._net_x(Xy_va)
                    lvx = self._gaussian_nll(pvx, Xn_va - base_xv_val, var_x)
                    val_loss = (lvy + (lvx / d_x)).item()
                self._net_x.train()
                self._net_y.train()
                self._net_x_var.train()
                self._net_y_var.train()
            else:
                val_loss = epoch_loss

            if self.verbose and epoch % 100 == 0:
                print(f'Epoch {epoch + 1}/{self.epochs} | Train NLL: {epoch_loss:.4f} | Val NLL: {val_loss:.4f}')

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if self.verbose and new_lr != current_lr:
                print(f'Epoch {epoch + 1}: Learning rate reduced from {current_lr} to {new_lr}')

            margin = adaptive_min_delta if adaptive_min_delta is not None else 0.0
            if val_loss + margin < best:
                best_train = epoch_loss
                best = val_loss
                wait = 0
                best_state = {
                    'net_x': {k: v.detach().cpu().clone() for k, v in self._net_x.state_dict().items()},
                    'net_y': {k: v.detach().cpu().clone() for k, v in self._net_y.state_dict().items()},
                    'net_x_var': {k: v.detach().cpu().clone() for k, v in self._net_x_var.state_dict().items()},
                    'net_y_var': {k: v.detach().cpu().clone() for k, v in self._net_y_var.state_dict().items()},
                }
                if use_val:
                    frac = self.min_delta_fraction if self.min_delta_fraction > 0.0 else 0.0
                    adaptive_min_delta = frac * abs(best)
                else:
                    adaptive_min_delta = 0.0
                self._current_min_delta = adaptive_min_delta
            else:
                wait += 1
                if self.early_stopping and wait >= self.patience:
                    if self.verbose:
                        print(
                            f'Early stopping at epoch {epoch + 1}. '
                            f'Best Val NLL: {best:.4f} (Train NLL: {best_train:.4f})'
                        )
                    break

        if best_state is not None:
            self._net_x.load_state_dict(best_state['net_x'])
            self._net_y.load_state_dict(best_state['net_y'])
            self._net_x_var.load_state_dict(best_state['net_x_var'])
            self._net_y_var.load_state_dict(best_state['net_y_var'])

        # temperature scaling on validation split (scalar adjustment of predicted variances)
        if use_val:
            with torch.inference_mode():
                if X_va.shape[0] > 0:
                    # stage 1 residuals (y)
                    var_y_t = self._var_y(X_va)
                    target_yv = y_va - base_yv_val
                    pred_yv = self._net_y(X_va)
                    resid_yv = target_yv - pred_yv
                    if resid_yv.numel() > 0:
                        temp_y_num = torch.sum(resid_yv.pow(2) / var_y_t)
                        temp_y_den = resid_yv.numel()
                        self._temp_y = (temp_y_num / max(temp_y_den, 1)).clamp(min=1e-6).item()

                    # stage 2 residuals (x)
                    var_x_t = self._var_x(Xy_va)
                    target_xv = Xn_va - base_xv_val
                    pred_xv = self._net_x(Xy_va)
                    resid_xv = target_xv - pred_xv
                    if resid_xv.numel() > 0 and var_x_t.numel() > 0:
                        temp_x_num = torch.sum(resid_xv.pow(2) / var_x_t)
                        temp_x_den = resid_xv.numel()
                        self._temp_x = (temp_x_num / max(temp_x_den, 1)).clamp(min=1e-6).item()

        # self._diagnosis_fitting(d_next, X_ctx, Y_next)

        self._net_x.eval()
        self._net_y.eval()
        self._net_x_var.eval()
        self._net_y_var.eval()

        self._x_mean_t = torch.as_tensor(self._x_mean, dtype=torch.float32, device=self.device)
        self._x_scale_t = torch.as_tensor(self._x_scale, dtype=torch.float32, device=self.device)
        self._xnext_mean_t = torch.as_tensor(self._xnext_mean, dtype=torch.float32, device=self.device)
        self._xnext_scale_t = torch.as_tensor(self._xnext_scale, dtype=torch.float32, device=self.device)
        self._y_mean_t = torch.as_tensor(self._y_mean, dtype=torch.float32, device=self.device)
        self._y_scale_t = torch.as_tensor(self._y_scale, dtype=torch.float32, device=self.device)

        self._d_x = d_x
        self._d_next = d_next
        self._fitted = True

        return self

    def sample_next(
        self,
        X_ctx: np.ndarray,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
        z: Optional[np.ndarray] = None,
        antithetic: bool = True,
    ) -> np.ndarray:
        if self._net_y is None or self._net_x is None or self._net_y_var is None or self._net_x_var is None:
            raise RuntimeError('Transition model is not fitted.')

        if rng is None:
            rng = np.random.default_rng()

        X_ctx = np.asarray(X_ctx, dtype=np.float32, order='C')
        N = X_ctx.shape[0]
        d_x = self._d_x
        d_next = self._d_next

        # ---- noise ----
        if z is None:
            m = (n_samples + 1) // 2 if antithetic else n_samples
            z0 = rng.standard_normal(size=(N, m, d_next)).astype(np.float32)
            if antithetic:
                cutoff = m if n_samples % 2 == 0 else m - 1
                z = np.concatenate([z0, -z0[:, :cutoff, :]], axis=1)
            else:
                z = z0
        else:
            z = np.asarray(z, dtype=np.float32)

        if z.shape != (N, n_samples, d_next):
            raise ValueError(f'z must have shape {(N, n_samples, d_next)} but got {z.shape}.')

        z_t = torch.from_numpy(z).to(self.device)
        X_t0 = torch.from_numpy(X_ctx).to(self.device)

        with torch.inference_mode():
            # standardize X once
            X_t = (X_t0 - self._x_mean_t) / self._x_scale_t  # (N, d_in)

            # ---- stage 1: y ----
            pred_y = self._net_y(X_t)  # (N,1)
            base_y = self._baseline(X_t, self._lin_coef_y_t, self._lin_intercept_y_t, pred_y)

            y_std_mu = pred_y + base_y
            y_mu = y_std_mu * self._y_scale_t + self._y_mean_t  # (N,1)

            var_y_std = self._var_y(X_t)
            var_y = var_y_std * (self._y_scale_t**2)
            zy = z_t[:, :, d_x : d_x + 1]  # (N,S,1)
            y_samples = y_mu.unsqueeze(1) + zy * torch.sqrt(var_y).unsqueeze(1)  # (N,S,1)

            # ---- stage 2: x | y ----
            if d_x > 0:
                S = n_samples
                X_rep = X_t.repeat_interleave(S, dim=0)  # (N*S, d_in)
                y_flat = y_samples.reshape(N * S, 1)  # (N*S, 1)
                y_std = (y_flat - self._y_mean_t) / self._y_scale_t  # (N*S, 1)
                Xy = torch.cat([X_rep, y_std], dim=1)  # (N*S, d_in+1)

                pred_x = self._net_x(Xy)  # (N*S, d_x)
                base_x = self._baseline(Xy, self._lin_coef_x_t, self._lin_intercept_x_t, pred_x)

                x_std_mu = pred_x + base_x
                x_mu = x_std_mu * self._xnext_scale_t + self._xnext_mean_t
                x_mu = x_mu.reshape(N, S, d_x)

                var_x_std = self._var_x(Xy)
                var_x = var_x_std * (self._xnext_scale_t**2)
                var_x = var_x.reshape(N, S, d_x)

                zx = z_t[:, :, :d_x]  # (N,S,d_x)
                x_samples = x_mu + zx * torch.sqrt(var_x)
            else:
                x_samples = torch.zeros((N, n_samples, 0), device=self.device)

            samples = torch.cat([x_samples, y_samples], dim=2)

        return samples.cpu().numpy().astype(float)

    def _reset_linear_baselines(self) -> None:
        self._lin_coef_x = None
        self._lin_intercept_x = None
        self._lin_coef_y = None
        self._lin_intercept_y = None
        self._lin_coef_x_t = None
        self._lin_intercept_x_t = None
        self._lin_coef_y_t = None
        self._lin_intercept_y_t = None

    def _fit_linear_baselines(
        self,
        X_std: np.ndarray,
        Xn_std: np.ndarray,
        y_std: np.ndarray,
        tr_idx: np.ndarray,
    ) -> None:
        if not self.use_residual_baseline:
            self._reset_linear_baselines()
            return

        if X_std.shape[1] == 0:
            coef_y = np.zeros((0, y_std.shape[1]), dtype=np.float32)
            intercept_y = np.asarray(y_std[tr_idx].mean(axis=0), dtype=np.float32).reshape(1, -1)
        else:
            lin_y = LinearRegression()
            lin_y.fit(X_std[tr_idx], y_std[tr_idx])
            coef_y = np.asarray(lin_y.coef_.T, dtype=np.float32)
            intercept_y = np.asarray(lin_y.intercept_.reshape(1, -1), dtype=np.float32)
        self._lin_coef_y = coef_y
        self._lin_intercept_y = intercept_y
        self._lin_coef_y_t = torch.as_tensor(self._lin_coef_y, device=self.device)
        self._lin_intercept_y_t = torch.as_tensor(self._lin_intercept_y, device=self.device)

        Xy_for_x = np.c_[X_std, y_std]
        if Xn_std.shape[1] == 0:
            coef_x = np.zeros((Xy_for_x.shape[1], 0), dtype=np.float32)
            intercept_x = np.empty((0,), dtype=np.float32)
        else:
            lin_x = LinearRegression()
            lin_x.fit(Xy_for_x[tr_idx], Xn_std[tr_idx])
            coef_x = np.asarray(lin_x.coef_.T, dtype=np.float32)
            intercept_x = np.asarray(lin_x.intercept_, dtype=np.float32)
        self._lin_coef_x = coef_x
        self._lin_intercept_x = intercept_x
        self._lin_coef_x_t = torch.as_tensor(self._lin_coef_x, device=self.device)
        self._lin_intercept_x_t = torch.as_tensor(self._lin_intercept_x, device=self.device)

    def _baseline(
        self,
        X: torch.Tensor,
        coef: Optional[torch.Tensor],
        intercept: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_residual_baseline and coef is not None and intercept is not None:
            return X @ coef + intercept
        return torch.zeros_like(like)

    # mean of x-next given X and y (standardized pipeline + baseline + mlp)
    @torch.inference_mode()
    def _mean_x(self, X_ctx: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError('Model not fit')
        d_x = self._d_x if self._d_x is not None else 0
        if d_x == 0:
            if y.ndim == 3:
                return np.zeros((y.shape[0], y.shape[1], 0), dtype=float)
            return np.zeros((X_ctx.shape[0], 0), dtype=float)
        X_std = (X_ctx - self._x_mean) / self._x_scale
        Xt = torch.from_numpy(X_std.astype(np.float32)).to(self.device)
        if y.ndim == 2:
            y_std = (y - self._y_mean) / self._y_scale
            y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(self.device)
            Xy_t = torch.cat([Xt, y_std_t], dim=1)

            resid_t = self._net_x(Xy_t)
            base_t = self._baseline(Xy_t, self._lin_coef_x_t, self._lin_intercept_x_t, resid_t)

            x_std_t = resid_t + base_t
            x_std = x_std_t.cpu().numpy()
            return x_std * self._xnext_scale + self._xnext_mean

        if y.ndim == 3:
            N, S, _ = y.shape
            X_rep_t = Xt.repeat_interleave(S, dim=0)
            y_flat = y.reshape(N * S, -1)
            y_std = (y_flat - self._y_mean) / self._y_scale
            y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(self.device)
            Xy_t = torch.cat([X_rep_t, y_std_t], dim=1)

            pred = self._net_x(Xy_t)
            resid_t = pred.reshape(N, S, -1)

            base = self._baseline(Xy_t, self._lin_coef_x_t, self._lin_intercept_x_t, pred)
            base_t = base.reshape(N, S, -1)

            x_std_t = resid_t + base_t
            x_std = x_std_t.cpu().numpy()
            return x_std * self._xnext_scale.reshape(1, 1, -1) + self._xnext_mean.reshape(1, 1, -1)

        raise ValueError(f'y must have ndim 2 or 3, but got y.ndim={y.ndim}.')

    # mean of y given X (standardized pipeline + baseline + mlp)
    @torch.inference_mode()
    def _mean_y(self, X_ctx: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError('Model not fit')
        X_std = (X_ctx - self._x_mean) / self._x_scale
        Xt = torch.from_numpy(X_std.astype(np.float32)).to(self.device)
        resid_t = self._net_y(Xt)
        base_t = self._baseline(Xt, self._lin_coef_y_t, self._lin_intercept_y_t, resid_t)

        y_std_t = resid_t + base_t
        y_std = y_std_t.cpu().numpy()
        return (y_std * self._y_scale + self._y_mean).reshape(-1, 1)

    def _mean(self, X_ctx: np.ndarray) -> np.ndarray:
        # unconditional mean [E X_{t+1}, E Y_t | X] where x uses E[y|X]
        my = self._mean_y(X_ctx)
        mx = self._mean_x(X_ctx, my)
        return np.concatenate([mx, my], axis=1)

    def _var_x(self, Xy: torch.Tensor) -> torch.Tensor:
        if self._net_x_var is None:
            raise RuntimeError('Variance network (x) not initialised.')
        var = F.softplus(self._net_x_var(Xy)) + self._var_eps
        if self._temp_x != 1.0:
            var = var * self._temp_x
        return var

    def _var_y(self, X: torch.Tensor) -> torch.Tensor:
        if self._net_y_var is None:
            raise RuntimeError('Variance network (y) not initialised.')
        var = F.softplus(self._net_y_var(X)) + self._var_eps
        if self._temp_y != 1.0:
            var = var * self._temp_y
        return var

    def _diagnosis_fitting(self, d_next: int, X_ctx: np.ndarray, Y_next: np.ndarray) -> None:
        try:
            d_x = d_next - 1
            X_chk = X_ctx
            Xnext_chk = Y_next[:, :d_x]
            y_chk = Y_next[:, -1:]

            X_std_chk = (X_chk - self._x_mean) / self._x_scale
            Xnext_std_chk = (Xnext_chk - self._xnext_mean) / self._xnext_scale
            y_std_chk = (y_chk - self._y_mean) / self._y_scale
            Xy_std_chk = np.c_[X_std_chk, y_std_chk]

            Xy_t_chk = torch.from_numpy(Xy_std_chk.astype(np.float32)).to(self.device)

            if self.use_residual_baseline and self._lin_coef_x is not None:
                lin_pred = Xy_std_chk @ self._lin_coef_x + self._lin_intercept_x
            else:
                lin_pred = np.zeros_like(Xnext_std_chk)

            self._net_x.eval()
            with torch.inference_mode():
                mlp_pred_t = self._net_x(Xy_t_chk)
                mlp_pred = mlp_pred_t.cpu().numpy()
            self._net_x.train()

            norm_lin = np.linalg.norm(lin_pred, axis=1).mean()
            norm_mlp = np.linalg.norm(mlp_pred, axis=1).mean()
            ratio = norm_mlp / (norm_lin + 1e-12)

            print('-' * 60)
            print(f'[Fit Diagnosis] MLP/Linear Ratio: {ratio:.5f}')
            print(f'  - Linear output norm: {norm_lin:.4f}')
            print(f'  - MLP output norm   : {norm_mlp:.4f}')
            if ratio < 0.01:
                print('  !! WARNING: MLP contribution is negligible. Model is effectively Linear.')
            else:
                print('  OK: MLP is contributing to the prediction.')
            print('-' * 60)
        except Exception as e:
            print(f'[Fit Diagnosis] Failed to run diagnosis: {e}')


class MLPNonlinearZILNTransition(TransitionModel):
    def __init__(
        self,
        hidden_dim: int = 16,
        n_layers: int = 3,
        lr: float = 5e-3,
        weight_decay: float = 1e-2,
        epochs: int = 2000,
        batch_size: int = 512,
        seed: Optional[int] = None,
        early_stopping: bool = True,
        patience: int = 50,
        min_delta_fraction: float = 1e-4,
        val_fraction: float = 0.1,
        grad_clip_norm: Optional[float] = 1.0,
        grad_clip_value: Optional[float] = None,
        use_residual_baseline: bool = False,
        verbose: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.early_stopping = early_stopping
        self.patience = max(1, patience)
        self.min_delta_fraction = max(0.0, min_delta_fraction)
        self._current_min_delta: Optional[float] = None
        self.val_fraction = val_fraction
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value
        self.use_residual_baseline = use_residual_baseline
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # linear baselines (optional)
        self._lin_coef_logy = None
        self._lin_intercept_logy = None
        self._lin_coef_logx = None
        self._lin_intercept_logx = None
        self._lin_coef_logy_t = None
        self._lin_intercept_logy_t = None
        self._lin_coef_logx_t = None
        self._lin_intercept_logx_t = None

        self._fitted: bool = False
        self._var_eps = 1e-4
        self._temp_x: float = 1.0
        self._temp_y: float = 1.0

        if seed is not None:
            torch.manual_seed(seed)

    @staticmethod
    def _mlp(d_in: int, d_out: int, hidden: int, n_layers: int, dropout: float = 0.3) -> nn.Sequential:
        layers: list[nn.Module] = []
        if n_layers == 1:
            layers.append(nn.Linear(d_in, d_out))
        else:
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.SiLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))

            layers.append(nn.Linear(hidden, d_out))

        return nn.Sequential(*layers)

    @staticmethod
    def _init_mlp_weights(net: nn.Module) -> None:
        for module in net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.01, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _fit_scaler(data: np.ndarray, tr_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if data.shape[1] == 0:
            empty = np.empty((0,), dtype=float)
            return data.copy(), empty, empty
        scaler = StandardScaler()
        scaler.fit(data[tr_idx])
        data_std = scaler.transform(data)
        return data_std, scaler.mean_, scaler.scale_

    @staticmethod
    def _ziln_nll_vec(
        logit_pi: torch.Tensor,
        mu_log: torch.Tensor,
        var_log: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if y.ndim != 2:
            raise ValueError('ZILN NLL expects y to be a 2D tensor of shape (B, D).')
        if y.shape[1] == 0:
            return torch.tensor(0.0, device=y.device)
        if torch.any(y < 0):
            raise ValueError('ZILN requires non-negative targets (found y < 0).')

        is_zero = y <= 0.0

        # -log(sigmoid(logit_pi)) and -log(1 - sigmoid(logit_pi))
        nll_zero = F.softplus(-logit_pi)
        nll_pos_mix = F.softplus(logit_pi)

        y_pos = y.clamp_min(EPS)
        logy = torch.log(y_pos)
        resid = logy - mu_log
        nll_logn = 0.5 * (torch.log(var_log) + resid.pow(2) / var_log) + logy

        nll = torch.where(is_zero, nll_zero, nll_pos_mix + nll_logn)
        return nll.sum(dim=-1).mean()

    def fit(self, X_ctx: np.ndarray, Y_next: np.ndarray) -> 'MLPNonlinearZILNTransition':
        N, d_in = X_ctx.shape
        d_next = Y_next.shape[1]
        d_x = d_next - 1
        d_x_safe = max(1, d_x)

        Xnext = Y_next[:, :d_x]
        y = Y_next[:, -1:]

        # ZILN requires non-negative targets
        if np.any(y < 0) or np.any(Xnext < 0):
            raise ValueError('MLPNonlinearZILNTransition requires all targets >= 0.')

        # reset temperature scaling
        self._temp_x = 1.0
        self._temp_y = 1.0
        adaptive_min_delta: Optional[float] = None
        self._current_min_delta = None

        # split train/val
        if self.early_stopping and N > 1 and self.val_fraction > 0.0:
            v = min(max(1, round(self.val_fraction * N)), N - 1)
            perm = self.rng.permutation(N)
            tr_idx = perm[: N - v]
            va_idx = perm[N - v :]
        else:
            tr_idx = np.arange(N, dtype=np.int64)
            va_idx = None
        if tr_idx.size == 0:
            tr_idx = np.arange(N, dtype=np.int64)
            va_idx = None

        # normalization (fit on train only)
        X_std, self._x_mean, self._x_scale = self._fit_scaler(X_ctx, tr_idx)
        y_std, self._y_mean, self._y_scale = self._fit_scaler(y, tr_idx)
        # keep for compatibility (not used by ZILN likelihood)
        Xn_std, self._xnext_mean, self._xnext_scale = self._fit_scaler(Xnext, tr_idx)

        # ---- log-space scalers (fit on train positives only) ----
        # y
        pos_tr_mask_y = y[tr_idx, 0] > 0.0
        self._y_has_pos = bool(np.any(pos_tr_mask_y))
        if self._y_has_pos:
            logy_tr = np.log(y[tr_idx[pos_tr_mask_y]])
            logy_scaler = StandardScaler()
            logy_scaler.fit(logy_tr)
            self._logy_mean = logy_scaler.mean_
            self._logy_scale = logy_scaler.scale_
        else:
            self._logy_mean = np.zeros((1,), dtype=float)
            self._logy_scale = np.ones((1,), dtype=float)

        # Xnext (per-dimension)
        if d_x > 0:
            logx_mean = np.zeros((d_x,), dtype=float)
            logx_scale = np.ones((d_x,), dtype=float)
            x_has_pos = np.zeros((d_x,), dtype=bool)
            for j in range(d_x):
                pos_mask = Xnext[tr_idx, j] > 0.0
                if np.any(pos_mask):
                    x_has_pos[j] = True
                    logx = np.log(Xnext[tr_idx[pos_mask], j : j + 1])
                    sc = StandardScaler()
                    sc.fit(logx)
                    logx_mean[j] = sc.mean_[0]
                    s = sc.scale_[0]
                    logx_scale[j] = s if s > 0.0 else 1.0
            self._logx_mean = logx_mean
            self._logx_scale = logx_scale
            self._x_has_pos = x_has_pos
        else:
            self._logx_mean = np.empty((0,), dtype=float)
            self._logx_scale = np.empty((0,), dtype=float)
            self._x_has_pos = np.empty((0,), dtype=bool)

        # linear residual baselines
        self._fit_linear_baselines(X_std, y_std, y, Xnext, tr_idx)

        # tensors
        X_t = torch.from_numpy(X_std.astype(np.float32)).to(self.device)
        y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(self.device)
        y_raw_t = torch.from_numpy(y.astype(np.float32)).to(self.device)
        x_raw_t = torch.from_numpy(Xnext.astype(np.float32)).to(self.device)
        Xy_full = torch.cat([X_t, y_std_t], dim=1)

        # torch scalers
        self._x_mean_t = torch.as_tensor(self._x_mean, dtype=torch.float32, device=self.device)
        self._x_scale_t = torch.as_tensor(self._x_scale, dtype=torch.float32, device=self.device)
        self._y_mean_t = torch.as_tensor(self._y_mean, dtype=torch.float32, device=self.device)
        self._y_scale_t = torch.as_tensor(self._y_scale, dtype=torch.float32, device=self.device)

        self._logy_mean_t = torch.as_tensor(self._logy_mean, dtype=torch.float32, device=self.device)
        self._logy_scale_t = torch.as_tensor(self._logy_scale, dtype=torch.float32, device=self.device)
        self._logx_mean_t = torch.as_tensor(self._logx_mean, dtype=torch.float32, device=self.device)
        self._logx_scale_t = torch.as_tensor(self._logx_scale, dtype=torch.float32, device=self.device)
        self._x_has_pos_t = torch.as_tensor(self._x_has_pos, dtype=torch.bool, device=self.device)

        # --- y networks (pi, mu, var) ---
        self._net_y_pi = self._mlp(d_in, 1, self.hidden_dim, self.n_layers).to(self.device)
        self._net_y = self._mlp(d_in, 1, self.hidden_dim, self.n_layers).to(self.device)
        self._net_y_var = self._mlp(d_in, 1, self.hidden_dim, self.n_layers).to(self.device)
        self._init_mlp_weights(self._net_y_pi)
        self._init_mlp_weights(self._net_y)
        self._init_mlp_weights(self._net_y_var)

        # --- x networks (pi, mu, var) ---
        in_x = d_in + 1
        self._net_x_pi = self._mlp(in_x, d_x, self.hidden_dim, self.n_layers).to(self.device)
        self._net_x = self._mlp(in_x, d_x, self.hidden_dim, self.n_layers).to(self.device)
        self._net_x_var = self._mlp(in_x, d_x, self.hidden_dim, self.n_layers).to(self.device)
        self._init_mlp_weights(self._net_x_pi)
        self._init_mlp_weights(self._net_x)
        self._init_mlp_weights(self._net_x_var)

        # optimizer: weight decay only on weights
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []

        def _collect_params(net: nn.Module) -> None:
            for _, p in net.named_parameters():
                (decay if p.dim() >= 2 else no_decay).append(p)

        _collect_params(self._net_y_pi)
        _collect_params(self._net_y)
        _collect_params(self._net_y_var)
        _collect_params(self._net_x_pi)
        _collect_params(self._net_x)
        _collect_params(self._net_x_var)

        groups = []
        if decay:
            groups.append({'params': decay, 'weight_decay': self.weight_decay})
        if no_decay:
            groups.append({'params': no_decay, 'weight_decay': 0.0})

        optimizer = optim.AdamW(groups, lr=self.lr, eps=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

        # validation tensors
        use_val = self.early_stopping and va_idx is not None and va_idx.size > 0
        if use_val:
            vi = torch.as_tensor(va_idx, dtype=torch.long, device=self.device)
            X_va = X_t.index_select(0, vi)
            y_std_va = y_std_t.index_select(0, vi)
            y_raw_va = y_raw_t.index_select(0, vi)
            x_raw_va = x_raw_t.index_select(0, vi)
            Xy_va = Xy_full.index_select(0, vi)
            with torch.inference_mode():
                base_mu_y_val = self._baseline(X_va, self._lin_coef_logy_t, self._lin_intercept_logy_t, y_std_va)
                like_x = torch.zeros((Xy_va.shape[0], d_x), device=self.device)
                base_mu_x_val = self._baseline(Xy_va, self._lin_coef_logx_t, self._lin_intercept_logx_t, like_x)

        # training loop
        best = np.inf
        wait = 0
        best_state: Optional[Mapping[str, Any]] = None

        params_to_clip: list[torch.nn.Parameter] = []
        params_to_clip += list(self._net_y_pi.parameters())
        params_to_clip += list(self._net_y.parameters())
        params_to_clip += list(self._net_y_var.parameters())
        params_to_clip += list(self._net_x_pi.parameters())
        params_to_clip += list(self._net_x.parameters())
        params_to_clip += list(self._net_x_var.parameters())

        warmup = 10
        for epoch in range(self.epochs):
            if tr_idx.size > 0:
                self.rng.shuffle(tr_idx)
            seen = 0
            epoch_loss = 0.0

            for start in range(0, tr_idx.size, self.batch_size):
                sl = tr_idx[start : start + self.batch_size]
                if sl.size == 0:
                    continue
                si = torch.as_tensor(sl, dtype=torch.long, device=self.device)

                xb = X_t.index_select(0, si)
                y_rawb = y_raw_t.index_select(0, si)
                x_rawb = x_raw_t.index_select(0, si)
                Xyb = Xy_full.index_select(0, si)

                optimizer.zero_grad(set_to_none=True)

                # ---- stage 1: X -> y (ZILN) ----
                logit_pi_y = self._net_y_pi(xb)
                mu_resid_std_y = self._net_y(xb)
                base_mu_y = self._baseline(xb, self._lin_coef_logy_t, self._lin_intercept_logy_t, mu_resid_std_y)
                mu_std_y = mu_resid_std_y + base_mu_y
                mu_log_y = mu_std_y * self._logy_scale_t + self._logy_mean_t

                var_std_y = self._var_y(xb)
                var_log_y = var_std_y * (self._logy_scale_t**2)

                nll_y = self._ziln_nll_vec(logit_pi_y, mu_log_y, var_log_y, y_rawb)

                # ---- stage 2: [X, y] -> Xnext (ZILN, teacher forcing) ----
                if d_x > 0:
                    logit_pi_x = self._net_x_pi(Xyb)
                    mu_resid_std_x = self._net_x(Xyb)
                    base_mu_x = self._baseline(
                        Xyb,
                        self._lin_coef_logx_t,
                        self._lin_intercept_logx_t,
                        mu_resid_std_x,
                    )
                    mu_std_x = mu_resid_std_x + base_mu_x
                    mu_log_x = mu_std_x * self._logx_scale_t + self._logx_mean_t

                    var_std_x = self._var_x(Xyb)
                    var_log_x = var_std_x * (self._logx_scale_t**2)

                    nll_x = self._ziln_nll_vec(logit_pi_x, mu_log_x, var_log_x, x_rawb)
                else:
                    nll_x = torch.tensor(0.0, device=self.device)

                loss = nll_y + (nll_x / d_x_safe)
                loss.backward()

                if self.grad_clip_norm is not None:
                    mx = self.grad_clip_norm if epoch >= warmup else max(self.grad_clip_norm, 10.0)
                    torch.nn.utils.clip_grad_norm_(params_to_clip, mx)
                if self.grad_clip_value is not None:
                    torch.nn.utils.clip_grad_value_(params_to_clip, self.grad_clip_value)

                optimizer.step()

                bsz = xb.shape[0]
                seen += bsz
                epoch_loss += loss.item() * bsz

            epoch_loss = epoch_loss / max(1, seen)

            # validation
            if use_val:
                self._net_y_pi.eval()
                self._net_y.eval()
                self._net_y_var.eval()
                self._net_x_pi.eval()
                self._net_x.eval()
                self._net_x_var.eval()

                with torch.inference_mode():
                    # y
                    logit_pi_y_v = self._net_y_pi(X_va)
                    mu_resid_std_y_v = self._net_y(X_va)
                    mu_std_y_v = mu_resid_std_y_v + base_mu_y_val
                    mu_log_y_v = mu_std_y_v * self._logy_scale_t + self._logy_mean_t
                    var_std_y_v = self._var_y(X_va)
                    var_log_y_v = var_std_y_v * (self._logy_scale_t**2)
                    lvy = self._ziln_nll_vec(logit_pi_y_v, mu_log_y_v, var_log_y_v, y_raw_va)

                    # x
                    if d_x > 0:
                        logit_pi_x_v = self._net_x_pi(Xy_va)
                        mu_resid_std_x_v = self._net_x(Xy_va)
                        mu_std_x_v = mu_resid_std_x_v + base_mu_x_val
                        mu_log_x_v = mu_std_x_v * self._logx_scale_t + self._logx_mean_t
                        var_std_x_v = self._var_x(Xy_va)
                        var_log_x_v = var_std_x_v * (self._logx_scale_t**2)
                        lvx = self._ziln_nll_vec(logit_pi_x_v, mu_log_x_v, var_log_x_v, x_raw_va)
                    else:
                        lvx = torch.tensor(0.0, device=self.device)

                    val_loss = (lvy + (lvx / d_x_safe)).item()

                self._net_y_pi.train()
                self._net_y.train()
                self._net_y_var.train()
                self._net_x_pi.train()
                self._net_x.train()
                self._net_x_var.train()
            else:
                val_loss = epoch_loss

            if self.verbose and epoch % 100 == 0:
                print(f'Epoch {epoch + 1}/{self.epochs} | Train NLL: {epoch_loss:.4f} | Val NLL: {val_loss:.4f}')

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if self.verbose and new_lr != current_lr:
                print(f'Epoch {epoch + 1}: Learning rate reduced from {current_lr} to {new_lr}')

            margin = adaptive_min_delta if adaptive_min_delta is not None else 0.0
            if val_loss + margin < best:
                best_train = epoch_loss
                best = val_loss
                wait = 0
                best_state = {
                    'net_y_pi': {k: v.detach().cpu().clone() for k, v in self._net_y_pi.state_dict().items()},
                    'net_y': {k: v.detach().cpu().clone() for k, v in self._net_y.state_dict().items()},
                    'net_y_var': {k: v.detach().cpu().clone() for k, v in self._net_y_var.state_dict().items()},
                    'net_x_pi': {k: v.detach().cpu().clone() for k, v in self._net_x_pi.state_dict().items()},
                    'net_x': {k: v.detach().cpu().clone() for k, v in self._net_x.state_dict().items()},
                    'net_x_var': {k: v.detach().cpu().clone() for k, v in self._net_x_var.state_dict().items()},
                }
                if use_val:
                    frac = self.min_delta_fraction if self.min_delta_fraction > 0.0 else 0.0
                    adaptive_min_delta = frac * abs(best)
                else:
                    adaptive_min_delta = 0.0
                self._current_min_delta = adaptive_min_delta
            else:
                wait += 1
                if self.early_stopping and wait >= self.patience:
                    if self.verbose:
                        print(
                            f'Early stopping at epoch {epoch + 1}. '
                            f'Best Val NLL: {best:.4f} (Train NLL: {best_train:.4f})'
                        )
                    break

        if best_state is not None:
            self._net_y_pi.load_state_dict(best_state['net_y_pi'])
            self._net_y.load_state_dict(best_state['net_y'])
            self._net_y_var.load_state_dict(best_state['net_y_var'])
            self._net_x_pi.load_state_dict(best_state['net_x_pi'])
            self._net_x.load_state_dict(best_state['net_x'])
            self._net_x_var.load_state_dict(best_state['net_x_var'])

        # switch to eval for temperature scaling / inference
        self._net_y_pi.eval()
        self._net_y.eval()
        self._net_y_var.eval()
        self._net_x_pi.eval()
        self._net_x.eval()
        self._net_x_var.eval()

        # temperature scaling on validation split (scalar adjustment of predicted variances)
        if use_val:
            with torch.inference_mode():
                if X_va.shape[0] > 0:
                    # --- y: standardized log-space residuals on positive samples only ---
                    y_pos_mask = y_raw_va > 0.0
                    if torch.any(y_pos_mask):
                        var_std_y_v = self._var_y(X_va)
                        pred_mu_std_y_v = self._net_y(X_va) + base_mu_y_val
                        logy_v = torch.log(y_raw_va.clamp_min(EPS))
                        target_logy_std_v = (logy_v - self._logy_mean_t) / self._logy_scale_t
                        resid_y_v = target_logy_std_v - pred_mu_std_y_v

                        resid_y_pos = resid_y_v[y_pos_mask]
                        var_y_pos = var_std_y_v[y_pos_mask]
                        if resid_y_pos.numel() > 0:
                            temp_y_num = torch.sum(resid_y_pos.pow(2) / var_y_pos)
                            temp_y_den = resid_y_pos.numel()
                            self._temp_y = (temp_y_num / max(temp_y_den, 1)).clamp(min=1e-6).item()

                    # --- x: standardized log-space residuals on positive entries only ---
                    if d_x > 0:
                        var_std_x_v = self._var_x(Xy_va)
                        pred_mu_std_x_v = self._net_x(Xy_va) + base_mu_x_val

                        logx_v = torch.log(x_raw_va.clamp_min(EPS))
                        target_logx_std_v = (logx_v - self._logx_mean_t) / self._logx_scale_t
                        resid_x_v = target_logx_std_v - pred_mu_std_x_v

                        x_pos_mask = x_raw_va > 0.0
                        if torch.any(x_pos_mask):
                            resid_x_pos = resid_x_v[x_pos_mask]
                            var_x_pos = var_std_x_v[x_pos_mask]
                            if resid_x_pos.numel() > 0:
                                temp_x_num = torch.sum(resid_x_pos.pow(2) / var_x_pos)
                                temp_x_den = resid_x_pos.numel()
                                self._temp_x = (temp_x_num / max(temp_x_den, 1)).clamp(min=1e-6).item()

        self._d_x = d_x
        self._d_next = d_next
        self._fitted = True
        return self

    def sample_next(
        self,
        X_ctx: np.ndarray,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
        z: Optional[np.ndarray] = None,
        antithetic: bool = True,
    ) -> np.ndarray:
        if (
            self._net_y_pi is None
            or self._net_y is None
            or self._net_y_var is None
            or self._net_x_pi is None
            or self._net_x is None
            or self._net_x_var is None
        ):
            raise RuntimeError('Transition model is not fitted.')

        if rng is None:
            rng = np.random.default_rng()

        X_ctx = np.asarray(X_ctx, dtype=np.float32, order='C')
        N = X_ctx.shape[0]
        d_x = self._d_x or 0
        d_next = self._d_next or 1

        # ---- Gaussian noise for log-space sampling ----
        if z is None:
            m = (n_samples + 1) // 2 if antithetic else n_samples
            z0 = rng.standard_normal(size=(N, m, d_next)).astype(np.float32)
            if antithetic:
                cutoff = m if n_samples % 2 == 0 else m - 1
                z = np.concatenate([z0, -z0[:, :cutoff, :]], axis=1)
            else:
                z = z0
        else:
            z = np.asarray(z, dtype=np.float32)

        if z.shape != (N, n_samples, d_next):
            raise ValueError(f'z must have shape {(N, n_samples, d_next)} but got {z.shape}.')

        # ---- mixture uniforms (antithetic) ----
        def _antithetic_uniform(shape: tuple[int, int, int]) -> np.ndarray:
            n0, s0, d0 = shape
            if not antithetic:
                return rng.random(size=(n0, s0, d0)).astype(np.float32)
            m0 = (s0 + 1) // 2
            u0 = rng.random(size=(n0, m0, d0)).astype(np.float32)
            cutoff0 = m0 if s0 % 2 == 0 else m0 - 1
            u = np.concatenate([u0, 1.0 - u0[:, :cutoff0, :]], axis=1)
            return u[:, :s0, :]

        u_y = _antithetic_uniform((N, n_samples, 1))
        u_x = _antithetic_uniform((N, n_samples, d_x)) if d_x > 0 else None

        z_t = torch.from_numpy(z).to(self.device)
        u_y_t = torch.from_numpy(u_y).to(self.device)
        u_x_t = torch.from_numpy(u_x).to(self.device) if u_x is not None else None
        X_t0 = torch.from_numpy(X_ctx).to(self.device)

        with torch.inference_mode():
            X_t = (X_t0 - self._x_mean_t) / self._x_scale_t

            # ---- stage 1: y ~ ZILN ----
            logit_pi_y = self._net_y_pi(X_t)
            pi_y = torch.sigmoid(logit_pi_y)

            if not self._y_has_pos:
                pi_y = torch.ones_like(pi_y)

            mu_resid_std_y = self._net_y(X_t)
            base_mu_y = self._baseline(X_t, self._lin_coef_logy_t, self._lin_intercept_logy_t, mu_resid_std_y)
            mu_std_y = mu_resid_std_y + base_mu_y
            mu_log_y = mu_std_y * self._logy_scale_t + self._logy_mean_t

            var_std_y = self._var_y(X_t)
            var_log_y = var_std_y * (self._logy_scale_t**2)

            zy = z_t[:, :, d_x : d_x + 1]
            logy_samples = mu_log_y.unsqueeze(1) + zy * torch.sqrt(var_log_y).unsqueeze(1)
            y_pos_samples = torch.exp(logy_samples)

            is_zero_y = u_y_t < pi_y.unsqueeze(1)
            y_samples = torch.where(is_zero_y, torch.zeros_like(y_pos_samples), y_pos_samples)

            # ---- stage 2: x | y ~ ZILN (per-dim) ----
            if d_x > 0:
                S = n_samples
                X_rep = X_t.repeat_interleave(S, dim=0)  # (N*S, d_in)
                y_flat = y_samples.reshape(N * S, 1)
                y_std = (y_flat - self._y_mean_t) / self._y_scale_t
                Xy = torch.cat([X_rep, y_std], dim=1)  # (N*S, d_in+1)

                logit_pi_x = self._net_x_pi(Xy)
                pi_x = torch.sigmoid(logit_pi_x)

                # force pi=1 for dims that never had positives in training
                if self._x_has_pos_t.numel() > 0:
                    pi_x = torch.where(self._x_has_pos_t.unsqueeze(0), pi_x, torch.ones_like(pi_x))

                mu_resid_std_x = self._net_x(Xy)
                base_mu_x = self._baseline(Xy, self._lin_coef_logx_t, self._lin_intercept_logx_t, mu_resid_std_x)
                mu_std_x = mu_resid_std_x + base_mu_x
                mu_log_x = mu_std_x * self._logx_scale_t + self._logx_mean_t

                var_std_x = self._var_x(Xy)
                var_log_x = var_std_x * (self._logx_scale_t**2)

                zx = z_t[:, :, :d_x].reshape(N * S, d_x)
                logx_samples = mu_log_x + zx * torch.sqrt(var_log_x)
                x_pos_samples = torch.exp(logx_samples)

                uxf = u_x_t.reshape(N * S, d_x)
                is_zero_x = uxf < pi_x
                x_samples = torch.where(is_zero_x, torch.zeros_like(x_pos_samples), x_pos_samples)
                x_samples = x_samples.reshape(N, S, d_x)
            else:
                x_samples = torch.zeros((N, n_samples, 0), device=self.device)

            samples = torch.cat([x_samples, y_samples], dim=2)

        return samples.cpu().numpy().astype(float)

    def _reset_linear_baselines(self) -> None:
        self._lin_coef_logy = None
        self._lin_intercept_logy = None
        self._lin_coef_logx = None
        self._lin_intercept_logx = None
        self._lin_coef_logy_t = None
        self._lin_intercept_logy_t = None
        self._lin_coef_logx_t = None
        self._lin_intercept_logx_t = None

    def _fit_linear_baselines(
        self,
        X_std: np.ndarray,
        y_std: np.ndarray,
        y_raw: np.ndarray,
        Xnext_raw: np.ndarray,
        tr_idx: np.ndarray,
    ) -> None:
        if not self.use_residual_baseline:
            self._reset_linear_baselines()
            return

        d_in = X_std.shape[1]
        d_x = Xnext_raw.shape[1]
        d_xy = d_in + 1  # [X_std, y_std]

        # ---- baseline for mu_log(y) (fit on positive y only) ----
        pos_tr_idx_y = tr_idx[y_raw[tr_idx, 0] > 0.0]
        if pos_tr_idx_y.size == 0:
            coef_logy = np.zeros((d_in, 1), dtype=np.float32)
            intercept_logy = np.zeros((1, 1), dtype=np.float32)
        else:
            logy = np.log(y_raw[pos_tr_idx_y])
            logy_std = (logy - self._logy_mean.reshape(1, 1)) / self._logy_scale.reshape(1, 1)
            if d_in == 0:
                coef_logy = np.zeros((0, 1), dtype=np.float32)
                intercept_logy = np.asarray(logy_std.mean(axis=0), dtype=np.float32).reshape(1, 1)
            else:
                lin_mu_y = LinearRegression()
                lin_mu_y.fit(X_std[pos_tr_idx_y], logy_std)
                coef_logy = np.asarray(lin_mu_y.coef_.T, dtype=np.float32)
                intercept_logy = np.asarray(lin_mu_y.intercept_.reshape(1, 1), dtype=np.float32)

        self._lin_coef_logy = coef_logy
        self._lin_intercept_logy = intercept_logy
        self._lin_coef_logy_t = torch.as_tensor(self._lin_coef_logy, device=self.device)
        self._lin_intercept_logy_t = torch.as_tensor(self._lin_intercept_logy, device=self.device)

        # ---- baseline for mu_log(Xnext) (fit per-dimension on positives only) ----
        Xy_for_x = np.c_[X_std, y_std]  # (N, d_in+1)
        if d_x == 0:
            coef_logx = np.zeros((d_xy, 0), dtype=np.float32)
            intercept_logx = np.zeros((1, 0), dtype=np.float32)
        else:
            coef_logx = np.zeros((d_xy, d_x), dtype=np.float32)
            intercept_logx = np.zeros((1, d_x), dtype=np.float32)
            for j in range(d_x):
                pos_mask = Xnext_raw[tr_idx, j] > 0.0
                pos_idx = tr_idx[pos_mask]
                if pos_idx.size == 0:
                    continue
                logx = np.log(Xnext_raw[pos_idx, j : j + 1])
                logx_std = (logx - self._logx_mean[j]) / self._logx_scale[j]
                lin_mu_x = LinearRegression()
                lin_mu_x.fit(Xy_for_x[pos_idx], logx_std)
                coef_logx[:, j : j + 1] = np.asarray(lin_mu_x.coef_.T, dtype=np.float32)
                intercept_logx[0, j] = lin_mu_x.intercept_[0]

        self._lin_coef_logx = coef_logx
        self._lin_intercept_logx = intercept_logx
        self._lin_coef_logx_t = torch.as_tensor(self._lin_coef_logx, device=self.device)
        self._lin_intercept_logx_t = torch.as_tensor(self._lin_intercept_logx, device=self.device)

    def _baseline(
        self,
        X: torch.Tensor,
        coef: Optional[torch.Tensor],
        intercept: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_residual_baseline and coef is not None and intercept is not None:
            return X @ coef + intercept
        return torch.zeros_like(like)

    @torch.inference_mode()
    def _mean_y(self, X_ctx: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError('Model not fit')
        X_std = (X_ctx - self._x_mean) / self._x_scale
        Xt = torch.from_numpy(X_std.astype(np.float32)).to(self.device)

        logit_pi = self._net_y_pi(Xt)
        pi = torch.sigmoid(logit_pi)
        if not self._y_has_pos:
            pi = torch.ones_like(pi)

        mu_resid_std = self._net_y(Xt)
        base_mu = self._baseline(Xt, self._lin_coef_logy_t, self._lin_intercept_logy_t, mu_resid_std)
        mu_std = mu_resid_std + base_mu
        mu_log = mu_std * self._logy_scale_t + self._logy_mean_t

        var_std = self._var_y(Xt)
        var_log = var_std * (self._logy_scale_t**2)

        mean_pos = torch.exp(mu_log + 0.5 * var_log)
        mean = (1.0 - pi) * mean_pos
        return mean.cpu().numpy().reshape(-1, 1)

    @torch.inference_mode()
    def _mean_x(self, X_ctx: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError('Model not fit')
        d_x = self._d_x or 0
        if d_x == 0:
            if y.ndim == 3:
                return np.zeros((y.shape[0], y.shape[1], 0), dtype=float)
            return np.zeros((X_ctx.shape[0], 0), dtype=float)

        X_std = (X_ctx - self._x_mean) / self._x_scale
        Xt = torch.from_numpy(X_std.astype(np.float32)).to(self.device)

        if y.ndim == 2:
            y_std = (y - self._y_mean) / self._y_scale
            y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(self.device)
            Xy_t = torch.cat([Xt, y_std_t], dim=1)

            logit_pi_x = self._net_x_pi(Xy_t)
            pi_x = torch.sigmoid(logit_pi_x)
            if self._x_has_pos_t.numel() > 0:
                pi_x = torch.where(self._x_has_pos_t.unsqueeze(0), pi_x, torch.ones_like(pi_x))

            mu_resid_std_x = self._net_x(Xy_t)
            base_mu_x = self._baseline(Xy_t, self._lin_coef_logx_t, self._lin_intercept_logx_t, mu_resid_std_x)
            mu_std_x = mu_resid_std_x + base_mu_x
            mu_log_x = mu_std_x * self._logx_scale_t + self._logx_mean_t

            var_std_x = self._var_x(Xy_t)
            var_log_x = var_std_x * (self._logx_scale_t**2)

            mean_pos = torch.exp(mu_log_x + 0.5 * var_log_x)
            mean = (1.0 - pi_x) * mean_pos
            return mean.cpu().numpy()

        if y.ndim == 3:
            N, S, _ = y.shape
            X_rep_t = Xt.repeat_interleave(S, dim=0)
            y_flat = y.reshape(N * S, -1)
            y_std = (y_flat - self._y_mean) / self._y_scale
            y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(self.device)
            Xy_t = torch.cat([X_rep_t, y_std_t], dim=1)

            logit_pi_x = self._net_x_pi(Xy_t)
            pi_x = torch.sigmoid(logit_pi_x)
            if self._x_has_pos_t.numel() > 0:
                pi_x = torch.where(self._x_has_pos_t.unsqueeze(0), pi_x, torch.ones_like(pi_x))

            mu_resid_std_x = self._net_x(Xy_t)
            base_mu_x = self._baseline(Xy_t, self._lin_coef_logx_t, self._lin_intercept_logx_t, mu_resid_std_x)
            mu_std_x = mu_resid_std_x + base_mu_x
            mu_log_x = mu_std_x * self._logx_scale_t + self._logx_mean_t

            var_std_x = self._var_x(Xy_t)
            var_log_x = var_std_x * (self._logx_scale_t**2)

            mean_pos = torch.exp(mu_log_x + 0.5 * var_log_x)
            mean = (1.0 - pi_x) * mean_pos
            return mean.reshape(N, S, d_x).cpu().numpy()

        raise ValueError(f'y must have ndim 2 or 3, but got y.ndim={y.ndim}.')

    def _mean(self, X_ctx: np.ndarray) -> np.ndarray:
        my = self._mean_y(X_ctx)
        mx = self._mean_x(X_ctx, my)
        return np.concatenate([mx, my], axis=1)

    def _var_x(self, Xy: torch.Tensor) -> torch.Tensor:
        if self._net_x_var is None:
            raise RuntimeError('Variance network (x) not initialised.')
        var = F.softplus(self._net_x_var(Xy)) + self._var_eps
        if self._temp_x != 1.0:
            var = var * self._temp_x
        return var

    def _var_y(self, X: torch.Tensor) -> torch.Tensor:
        if self._net_y_var is None:
            raise RuntimeError('Variance network (y) not initialised.')
        var = F.softplus(self._net_y_var(X)) + self._var_eps
        if self._temp_y != 1.0:
            var = var * self._temp_y
        return var


class VARTransition(TransitionModel):
    def __init__(self) -> None:
        self._B: Optional[np.ndarray] = None  # Shape (d_exog, d_next)
        self._c: Optional[np.ndarray] = None  # Shape (1, d_next)
        self._Sigma: Optional[np.ndarray] = None  # Shape (d_next, d_next)
        self._d_exog: Optional[int] = None
        self._d_next: Optional[int] = None

    def fit(self, X_exog: np.ndarray, Y_next: np.ndarray) -> 'VARTransition':
        if X_exog.ndim != 2 or Y_next.ndim != 2:
            raise ValueError('X_exog and Y_next must be 2D arrays.')

        N, d_exog = X_exog.shape
        d_next = Y_next.shape[1]
        if Y_next.shape[0] != N:
            raise ValueError('The number of rows in X_exog and Y_next must match.')

        model = VAR(endog=Y_next.astype(float), exog=X_exog)
        res = model.fit(maxlags=0, trend='c')

        if res.params.shape[0] != 1 + d_exog or res.params.shape[1] != d_next:
            raise RuntimeError('Unexpected shape of VAR parameters.')
        c = res.params[0:1, :]  # (1, d_next)
        B = res.params[1:, :]  # (d_exog, d_next)
        Sigma = _ensure_pos_def(res.sigma_u)

        self._B = B
        self._c = c
        self._Sigma = Sigma
        self._d_next = d_next
        self._d_exog = d_exog
        return self

    def sample_next(
        self,
        X_exog: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        z: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._B is None or self._c is None or self._Sigma is None:
            raise RuntimeError('Model has not been fitted. Please call `fit` first.')
        if self._d_next is None or self._d_exog is None:
            raise RuntimeError('Model parameters are incomplete. Please call `fit` first.')
        if X_exog.ndim != 2:
            raise ValueError('X_exog must be a 2D array.')

        if X_exog.shape[1] != self._d_exog:
            raise ValueError('The exogenous dimension does not match the one during training.')

        mean = (X_exog @ self._B) + self._c  # (N, d_next)
        try:
            L = np.linalg.cholesky(self._Sigma)  # (d_next, d_next)
        except np.linalg.LinAlgError:
            Sigma = _ensure_pos_def(self._Sigma)
            self._Sigma = Sigma
            L = np.linalg.cholesky(Sigma)

        if z is None:
            z = rng.normal(size=(X_exog.shape[0], n_samples, self._d_next))
        else:
            z = np.asarray(z, dtype=float)
            if z.shape != (X_exog.shape[0], n_samples, self._d_next):
                raise ValueError(f'z must have shape {(X_exog.shape[0], n_samples, self._d_next)} but got {z.shape}.')
        noise = z @ L.T
        samples = mean[:, None, :] + noise
        return np.asarray(samples, dtype=float)
