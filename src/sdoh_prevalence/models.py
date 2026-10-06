"""Modeling: regularized regression on collinear social-need predictors.

Two things this module is careful about, both of which matter more than any
hyperparameter:

* **Scaling inside the fold.** Standardization is fitted on the training split
  only, via a Pipeline. Scaling the whole dataset first leaks the test folds'
  means and standard deviations into training.
* **Grouped cross-validation.** Counties in the same state share BRFSS sampling
  design and state policy, so a random split puts near-siblings on both sides of
  the divide and flatters the model. ``GroupKFold`` on state keeps whole states
  together, which is the honest test of "would this transfer to a state I have
  not seen".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .config import ColumnMap
from .hyperparameters import alpha_grid

__all__ = [
    "XY",
    "make_xy",
    "make_scarce_regime",
    "build_pipeline",
    "grouped_cv",
    "cross_validated_rmse",
    "tune_alpha",
    "regularization_path",
    "compare_models",
    "correlation_table",
    "sgd_learning_curve",
]


@dataclass
class XY:
    """Feature matrix, target, grouping vector, and feature names."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.y)


def make_xy(df: pd.DataFrame, columns: ColumnMap = cfg.DEFAULT_COLUMNS) -> XY:
    """Split the analytic frame into X, y and the grouping column."""
    features = [c for c in columns.predictors if c in df.columns]
    if not features:
        raise ValueError(f"None of {columns.predictors} are in the dataframe")
    return XY(
        X=df[features].astype(float),
        y=df[columns.outcome].astype(float),
        groups=df[columns.group],
        feature_names=features,
    )


def make_scarce_regime(
    df: pd.DataFrame,
    columns: ColumnMap = cfg.DEFAULT_COLUMNS,
    n_counties: int = 120,
    degree: int = 2,
    random_state: int = cfg.RANDOM_STATE,
) -> XY:
    """Build the regime where regularization strength actually decides the model.

    With ~3,100 counties and 8 predictors, a penalty barely changes anything --
    there is simply too much data to overfit 8 coefficients. That is the honest
    result on the full file, and it is also why so many tutorials demonstrate
    regularization on data where it does nothing visible.

    Regularization earns its keep when the number of parameters approaches the
    number of observations. This function creates that situation the way it
    actually arises in applied work: restrict the analysis to a small set of
    counties (one state, or a rural subset), then add interaction and squared
    terms because you suspect social needs compound rather than simply add. Eight
    predictors at degree 2 become 44, against ~120 counties.
    """
    from sklearn.preprocessing import PolynomialFeatures

    features = [c for c in columns.predictors if c in df.columns]
    sample = df.sample(
        n=min(n_counties, len(df)), random_state=random_state
    ).reset_index(drop=True)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    expanded = poly.fit_transform(sample[features].astype(float))
    names = list(poly.get_feature_names_out(features))

    return XY(
        X=pd.DataFrame(expanded, columns=names),
        y=sample[columns.outcome].astype(float),
        groups=sample[columns.group],
        feature_names=names,
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Standardize then fit -- scaler parameters learned per training fold."""
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def grouped_cv(n_splits: int = 5) -> GroupKFold:
    return GroupKFold(n_splits=n_splits)


def cross_validated_rmse(
    estimator: BaseEstimator, data: XY, n_splits: int = 5
) -> tuple[float, float]:
    """Return (mean RMSE, sd of RMSE) across state-grouped folds."""
    scores = cross_val_score(
        build_pipeline(estimator),
        data.X,
        data.y,
        groups=data.groups,
        cv=grouped_cv(n_splits),
        scoring="neg_root_mean_squared_error",
    )
    rmse = -scores
    return float(rmse.mean()), float(rmse.std())


def tune_alpha(
    data: XY,
    alphas: Sequence[float] | np.ndarray | None = None,
    model: str = "ridge",
    n_splits: int = 5,
) -> pd.DataFrame:
    """Score every regularization strength with grouped CV.

    Returns a frame of ``alpha, rmse_mean, rmse_sd`` sorted by alpha. The best
    alpha is the one that minimizes ``rmse_mean``; a defensible alternative is
    the largest alpha within one standard error of that minimum, which buys a
    simpler model for almost no accuracy -- the "one-standard-error rule".
    """
    alphas = np.asarray(alphas if alphas is not None else alpha_grid())
    rows = []
    for alpha in alphas:
        est = Ridge(alpha=alpha) if model == "ridge" else Lasso(alpha=alpha, max_iter=20_000)
        mean, sd = cross_validated_rmse(est, data, n_splits=n_splits)
        rows.append({"alpha": float(alpha), "rmse_mean": mean, "rmse_sd": sd})
    out = pd.DataFrame(rows)

    best = out.loc[out["rmse_mean"].idxmin()]
    threshold = best["rmse_mean"] + best["rmse_sd"] / np.sqrt(n_splits)
    within = out[out["rmse_mean"] <= threshold]
    out.attrs["best_alpha"] = float(best["alpha"])
    out.attrs["best_rmse"] = float(best["rmse_mean"])
    out.attrs["one_se_alpha"] = float(within["alpha"].max())
    return out


def regularization_path(
    data: XY,
    alphas: Sequence[float] | np.ndarray | None = None,
    model: str = "ridge",
) -> pd.DataFrame:
    """Coefficients (on the standardized scale) at each regularization strength.

    Read the resulting frame as a story about collinearity: at low alpha the
    correlated social-need predictors trade large positive and negative
    coefficients that cancel out; as alpha rises they shrink toward a shared,
    stable, interpretable signal. Lasso additionally drives some to exactly
    zero -- which is variable *selection*, and choosing which collinear cousin
    survives is a decision the data cannot make for you.
    """
    alphas = np.asarray(alphas if alphas is not None else alpha_grid())
    rows = []
    for alpha in alphas:
        est = Ridge(alpha=alpha) if model == "ridge" else Lasso(alpha=alpha, max_iter=20_000)
        pipe = build_pipeline(est).fit(data.X, data.y)
        coefs = pipe.named_steps["model"].coef_
        rows.append({"alpha": float(alpha), **dict(zip(data.feature_names, coefs, strict=True))})
    out = pd.DataFrame(rows)
    out.attrs["model"] = model
    return out


def compare_models(data: XY, n_splits: int = 5) -> pd.DataFrame:
    """Grouped-CV RMSE for OLS and three tuned penalized fits."""
    ridge = tune_alpha(data, model="ridge", n_splits=n_splits)
    lasso = tune_alpha(data, model="lasso", n_splits=n_splits)

    candidates: dict[str, BaseEstimator] = {
        "OLS (no penalty)": LinearRegression(),
        f"Ridge (alpha={ridge.attrs['best_alpha']:.3g})": Ridge(
            alpha=ridge.attrs["best_alpha"]
        ),
        f"Lasso (alpha={lasso.attrs['best_alpha']:.3g})": Lasso(
            alpha=lasso.attrs["best_alpha"], max_iter=20_000
        ),
        "ElasticNet (alpha=0.1, l1=0.5)": ElasticNet(
            alpha=0.1, l1_ratio=0.5, max_iter=20_000
        ),
    }

    rows = []
    for name, est in candidates.items():
        mean, sd = cross_validated_rmse(est, data, n_splits=n_splits)
        fitted = build_pipeline(est).fit(data.X, data.y)
        coefs = getattr(fitted.named_steps["model"], "coef_", np.zeros(len(data.X.columns)))
        rows.append(
            {
                "model": name,
                "cv_rmse": round(mean, 4),
                "cv_rmse_sd": round(sd, 4),
                "in_sample_r2": round(r2_score(data.y, fitted.predict(data.X)), 4),
                "n_nonzero_coefs": int(np.sum(np.abs(coefs) > 1e-8)),
                "coef_l2_norm": round(float(np.linalg.norm(coefs)), 4),
            }
        )
    return pd.DataFrame(rows)


def correlation_table(data: XY) -> pd.DataFrame:
    """Predictor correlation matrix, labelled -- evidence for the penalty."""
    corr = data.X.corr()
    labels = [cfg.MEASURE_LABELS.get(c, c) for c in corr.columns]
    corr.index = labels
    corr.columns = labels
    return corr.round(2)


def sgd_learning_curve(
    data: XY,
    learning_rates: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    epochs: int = 60,
    random_state: int = cfg.RANDOM_STATE,
) -> pd.DataFrame:
    """Train the same model by SGD at several learning rates, epoch by epoch.

    This is where learning rate stops being abstract: too small and the loss is
    still falling when the epochs run out; too large and it diverges outright.
    Returns tidy ``learning_rate, epoch, train_rmse`` rows (NaN once diverged).
    """
    scaler = StandardScaler().fit(data.X)
    X = scaler.transform(data.X)
    y = data.y.to_numpy()

    rows = []
    for lr in learning_rates:
        model = SGDRegressor(
            learning_rate="constant",
            eta0=lr,
            penalty=None,
            max_iter=1,
            tol=None,
            warm_start=True,
            random_state=random_state,
        )
        for epoch in range(1, epochs + 1):
            with np.errstate(all="ignore"):
                model.fit(X, y)
                pred = model.predict(X)
            rmse = (
                float(np.sqrt(mean_squared_error(y, pred)))
                if np.all(np.isfinite(pred))
                else float("nan")
            )
            rows.append({"learning_rate": lr, "epoch": epoch, "train_rmse": rmse})
            if not np.isfinite(rmse):
                break
    return pd.DataFrame(rows)
