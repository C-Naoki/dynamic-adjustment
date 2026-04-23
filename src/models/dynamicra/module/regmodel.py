from typing import Any, Optional, Union

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor as SklearnRF


class RegressionModel:

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegressionModel':  # pragma: no cover
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def get_regression_factory(model_name: str, seed: Optional[int] = None) -> RegressionModel:
    if model_name == 'ols':
        return OLSRegressor()
    elif model_name == 'gbr':
        return GradientBoostingRegressor(random_state=seed)
    elif model_name == 'rf':
        return RandomForestRegressor(random_state=seed)
    else:
        raise ValueError(f'Unknown regression model name: {model_name}')


class OLSRegressor(RegressionModel):

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OLSRegressor':
        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.intercept_ + X @ self.coef_)


class GradientBoostingRegressor(RegressionModel):

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        min_samples_leaf: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 1.0,
        random_state: Optional[int] = None,
        n_jobs: int = 6,
        **kwargs: Any,
    ) -> None:
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_samples_leaf,
            subsample=subsample,
            subsample_freq=1,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type='gain',
            verbose=-1,
            **kwargs,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'GradientBoostingRegressor':
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.ndim != 1 or sample_weight.shape[0] != X.shape[0]:
                raise ValueError('sample_weight must be 1D and match X.')
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X))


class RandomForestRegressor(RegressionModel):

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_features: Union[str, float] = 1.0,
        max_samples: Optional[float] = None,
        random_state: Optional[int] = None,
        n_jobs: int = 6,
        **kwargs: Any,
    ) -> None:
        self.model = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'RandomForestRegressor':

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight).ravel()
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X))
