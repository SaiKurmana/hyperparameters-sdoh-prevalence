"""The arithmetic lessons, checked with tolerances instead of float equality."""

import math

import pytest

from sdoh_prevalence.hyperparameters import (
    add_absolute,
    alpha_grid,
    as_scientific,
    cosine_decay,
    decay_learning_rate,
    exponential_decay,
    scale_relative,
    schedule_history,
    simulate_loss_adjustments,
    step_decay,
)


def test_decay_matches_the_exercise():
    assert decay_learning_rate(0.01) == pytest.approx(0.01 * 0.9)
    assert decay_learning_rate(1e-3) == pytest.approx(9e-4)


def test_decay_rejects_out_of_range():
    with pytest.raises(ValueError):
        decay_learning_rate(0.01, decay=1.5)


def test_absolute_vs_relative():
    assert add_absolute(0.8, 0.02) == pytest.approx(0.82)
    assert scale_relative(0.82, 0.05) == pytest.approx(0.82 * 1.05)
    assert scale_relative(0.82, -0.05) == pytest.approx(0.82 * 0.95)


def test_loss_trajectory_reproduces_the_graders():
    steps = simulate_loss_adjustments()
    assert [label for label, _ in steps] == [
        "initial",
        "+ noisy small-county estimates",
        "+ overfitting",
        "- L2 regularization",
        "- more counties",
    ]
    assert steps[-1][1] == pytest.approx((0.8 + 0.02) * 1.05 * 0.95 - 0.05)


def test_percentage_changes_do_not_cancel():
    """+5% then -5% is a net decrease, not a round trip."""
    steps = dict(simulate_loss_adjustments())
    assert steps["- L2 regularization"] < steps["+ noisy small-county estimates"]
    assert not math.isclose(
        steps["- L2 regularization"], steps["+ noisy small-county estimates"]
    )


def test_step_decay_is_a_staircase():
    assert step_decay(1e-3, 0, drop=0.5, every=10) == pytest.approx(1e-3)
    assert step_decay(1e-3, 9, drop=0.5, every=10) == pytest.approx(1e-3)
    assert step_decay(1e-3, 10, drop=0.5, every=10) == pytest.approx(5e-4)
    assert step_decay(1e-3, 20, drop=0.5, every=10) == pytest.approx(2.5e-4)


def test_exponential_decay_never_reaches_zero():
    remaining = exponential_decay(1e-3, 20, decay=0.10) / 1e-3
    assert remaining == pytest.approx(0.9**20)
    assert 0.10 < remaining < 0.15  # ~12%, not 0 and not 1 - 20*0.1


def test_cosine_decay_endpoints():
    assert cosine_decay(1e-3, 0, 30) == pytest.approx(1e-3)
    assert cosine_decay(1e-3, 30, 30) == pytest.approx(0.0, abs=1e-12)
    assert cosine_decay(1e-3, 15, 30) == pytest.approx(5e-4)


def test_schedules_are_monotone_decreasing():
    for schedule in (
        lambda e: exponential_decay(1e-3, e, 0.1),
        lambda e: cosine_decay(1e-3, e, 30),
    ):
        history = schedule_history(schedule, epochs=30)
        assert all(b <= a + 1e-15 for a, b in zip(history, history[1:], strict=False))


def test_alpha_grid_is_log_spaced():
    grid = alpha_grid(1e-3, 1e3, n=7)
    assert grid[0] == pytest.approx(1e-3)
    assert grid[-1] == pytest.approx(1e3)
    ratios = grid[1:] / grid[:-1]
    assert ratios.std() == pytest.approx(0.0, abs=1e-9)


def test_as_scientific():
    assert as_scientific(0.001) == "1.0e-03"
    assert as_scientific(9e-4) == "9.0e-04"
