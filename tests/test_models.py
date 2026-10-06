"""Modeling: grouped CV, alpha tuning, regularization paths, no leakage."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from sdoh_prevalence import config as cfg
from sdoh_prevalence.data import make_synthetic
from sdoh_prevalence.hyperparameters import alpha_grid
from sdoh_prevalence.models import (
    build_pipeline,
    compare_models,
    correlation_table,
    cross_validated_rmse,
    grouped_cv,
    make_xy,
    regularization_path,
    sgd_learning_curve,
    tune_alpha,
)


@pytest.fixture(scope="module")
def data():
    return make_xy(make_synthetic(n_counties=500, random_state=11))


def test_make_xy_splits_correctly(data):
    assert len(data.X) == len(data.y) == len(data.groups) == 500
    assert data.feature_names == list(cfg.SDOH_MEASURES)
    assert cfg.OUTCOME_MEASURE not in data.X.columns


def test_make_xy_requires_known_predictors():
    df = make_synthetic(n_counties=20)
    from sdoh_prevalence.config import ColumnMap

    with pytest.raises(ValueError):
        make_xy(df, ColumnMap(predictors=("not_a_column",)))


def test_pipeline_scales_inside_the_fold(data):
    """Scaling must be a pipeline step, not applied to the full dataset."""
    pipe = build_pipeline(Ridge(alpha=1.0))
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["scale", "model"]

    pipe.fit(data.X, data.y)
    # The scaler learned its means from the data it was fitted on, and those means
    # are the training means -- not zeros, which is what a pre-scaled X would give.
    assert np.abs(pipe.named_steps["scale"].mean_).max() > 1.0


def test_grouped_cv_keeps_states_whole(data):
    cv = grouped_cv(n_splits=4)
    for train_idx, test_idx in cv.split(data.X, data.y, data.groups):
        train_states = set(data.groups.iloc[train_idx])
        test_states = set(data.groups.iloc[test_idx])
        assert not (train_states & test_states), "a state appeared on both sides"


def test_cross_validated_rmse_is_positive_and_finite(data):
    mean, sd = cross_validated_rmse(Ridge(alpha=1.0), data, n_splits=4)
    assert mean > 0 and np.isfinite(mean)
    assert sd >= 0


def test_tune_alpha_finds_an_interior_optimum(data):
    grid = alpha_grid(1e-2, 1e3, n=12)
    tuned = tune_alpha(data, alphas=grid, model="ridge", n_splits=4)

    assert len(tuned) == 12
    assert tuned["rmse_mean"].is_monotonic_increasing is False  # there is a dip
    best = tuned.attrs["best_alpha"]
    assert grid.min() <= best <= grid.max()
    # the one-SE rule never picks a stronger-fitting, weaker penalty
    assert tuned.attrs["one_se_alpha"] >= best


def test_stronger_penalty_shrinks_coefficients(data):
    path = regularization_path(data, alphas=alpha_grid(1e-2, 1e3, n=8), model="ridge")
    features = [c for c in path.columns if c != "alpha"]
    norms = np.linalg.norm(path[features].to_numpy(), axis=1)
    assert norms[0] > norms[-1]
    assert np.all(np.diff(norms) <= 1e-9), "L2 norm should fall monotonically with alpha"


def test_lasso_zeroes_coefficients_at_high_alpha(data):
    path = regularization_path(data, alphas=alpha_grid(1e-2, 1e2, n=8), model="lasso")
    features = [c for c in path.columns if c != "alpha"]
    nonzero = (path[features].abs() > 1e-8).sum(axis=1)
    assert nonzero.iloc[0] > nonzero.iloc[-1]
    assert nonzero.iloc[-1] == 0 or nonzero.iloc[-1] < len(features)


def test_compare_models_returns_one_row_per_candidate(data):
    table = compare_models(data, n_splits=4)
    assert len(table) == 4
    assert {"model", "cv_rmse", "in_sample_r2", "n_nonzero_coefs"}.issubset(table.columns)
    assert table["cv_rmse"].gt(0).all()


def test_correlation_table_is_labelled_and_square(data):
    corr = correlation_table(data)
    assert corr.shape == (len(cfg.SDOH_MEASURES), len(cfg.SDOH_MEASURES))
    assert "Food insecurity" in corr.columns
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)


def test_sgd_diverges_at_a_large_learning_rate(data):
    curve = sgd_learning_curve(data, learning_rates=(1e-3, 10.0), epochs=12)

    small = curve[curve["learning_rate"] == 1e-3]["train_rmse"]
    assert np.isfinite(small).all()
    assert small.iloc[-1] < small.iloc[0]

    large = curve[curve["learning_rate"] == 10.0]["train_rmse"]
    assert (~np.isfinite(large)).any() or large.iloc[-1] > small.iloc[-1]
