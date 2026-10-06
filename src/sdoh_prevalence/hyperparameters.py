"""Hyperparameter arithmetic and schedules, in population-health terms.

The functions here are the lesson content from ``scripts/01_assignment_operators.py``
and ``scripts/02_learning_rate_schedules.py``, factored out so they can be tested
and reused instead of living only inside a script.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

__all__ = [
    "decay_learning_rate",
    "add_absolute",
    "scale_relative",
    "simulate_loss_adjustments",
    "step_decay",
    "exponential_decay",
    "cosine_decay",
    "schedule_history",
    "as_scientific",
    "alpha_grid",
]


# --------------------------------------------------------------------------- #
# The assignment-operator primitives
# --------------------------------------------------------------------------- #


def decay_learning_rate(learning_rate: float, decay: float = 0.10) -> float:
    """Decrease ``learning_rate`` by ``decay`` (a fraction, not a percent).

    ``decay_learning_rate(0.01)`` -> ``0.009``. The multiplicative form
    (``lr *= 0.9``) is what every step-decay scheduler does under the hood.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0.0, 1.0)")
    return learning_rate * (1.0 - decay)


def add_absolute(value: float, amount: float) -> float:
    """``value += amount`` -- for changes stated in the metric's own units."""
    return value + amount


def scale_relative(value: float, pct_change: float) -> float:
    """``value *= (1 + pct_change)`` -- for changes stated as percentages.

    ``pct_change=0.05`` is a 5% increase; ``-0.05`` is a 5% decrease.
    """
    return value * (1.0 + pct_change)


def simulate_loss_adjustments(
    initial_loss: float = 0.8,
    noise: float = 0.02,
    overfitting_pct: float = 0.05,
    regularization_pct: float = 0.05,
    generalization_gain: float = 0.05,
) -> list[tuple[str, float]]:
    """Replay the four validation-loss adjustments as a trajectory.

    In this project's terms the loss is mean squared error on held-out counties,
    and the four steps are the story of a modeling session:

    1. noisier BRFSS estimates in small-population counties push error up,
    2. the model starts memorizing the training counties (overfitting),
    3. an L2 penalty pulls it back,
    4. adding more counties improves generalization.

    Returns ``[(label, value), ...]`` including the starting value.
    """
    steps: list[tuple[str, float]] = [("initial", initial_loss)]

    loss = initial_loss
    loss += noise
    steps.append(("+ noisy small-county estimates", loss))

    loss *= 1.0 + overfitting_pct
    steps.append(("+ overfitting", loss))

    loss *= 1.0 - regularization_pct
    steps.append(("- L2 regularization", loss))

    loss -= generalization_gain
    steps.append(("- more counties", loss))

    return steps


# --------------------------------------------------------------------------- #
# Learning rate schedules
# --------------------------------------------------------------------------- #


def step_decay(
    initial_lr: float, epoch: int, drop: float = 0.5, every: int = 10
) -> float:
    """Multiply the rate by ``drop`` every ``every`` epochs (staircase)."""
    if every <= 0:
        raise ValueError("every must be positive")
    return initial_lr * (drop ** (epoch // every))


def exponential_decay(initial_lr: float, epoch: int, decay: float = 0.10) -> float:
    """Smooth exponential decay: the compounded form of ``lr *= (1 - decay)``."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0.0, 1.0)")
    return initial_lr * (1.0 - decay) ** epoch


def cosine_decay(initial_lr: float, epoch: int, total_epochs: int) -> float:
    """Cosine annealing from ``initial_lr`` to 0 across ``total_epochs``."""
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    progress = min(epoch / total_epochs, 1.0)
    return initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def schedule_history(
    schedule: Callable[[int], float], epochs: int = 30
) -> list[float]:
    """Evaluate a 1-arg schedule at epochs ``0..epochs`` inclusive."""
    return [schedule(e) for e in range(epochs + 1)]


def as_scientific(value: float, precision: int = 1) -> str:
    """Render a small value in scientific notation: ``0.001`` -> ``1.0e-03``."""
    return f"{value:.{precision}e}"


# --------------------------------------------------------------------------- #
# Regularization strength
# --------------------------------------------------------------------------- #


def alpha_grid(low: float = 1e-3, high: float = 1e3, n: int = 60) -> np.ndarray:
    """Log-spaced grid of regularization strengths.

    Always search penalty strength on a log scale: the difference between
    ``alpha=0.01`` and ``alpha=0.1`` matters far more than between ``100`` and
    ``100.09``. A linear grid spends nearly all its budget in a range where
    nothing changes.
    """
    if low <= 0 or high <= low:
        raise ValueError("require 0 < low < high")
    return np.logspace(np.log10(low), np.log10(high), n)
