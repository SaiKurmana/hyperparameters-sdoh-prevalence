"""Figures for the exercise scripts.

Design rules applied throughout (worth knowing, since they are not matplotlib
defaults):

* **Categorical hues in fixed order, never cycled.** Series identity is assigned
  from a validated eight-slot palette; a ninth series would fold into "other"
  rather than invent a hue.
* **Ordered quantities get a single-hue ramp, not categorical colors.** Learning
  rates are ordered, so they read light-to-dark blue -- the reader can see which
  curve is the biggest rate without consulting the legend.
* **Correlations are diverging**: two opposed hues with a neutral gray midpoint,
  so zero looks like nothing and sign is visible at a glance.
* **Direct labels at the end of lines**, because three of these hues sit below
  3:1 contrast on a white surface and colour alone should never carry meaning.
* Recessive grid, no top/right spines, values also available as DataFrames so
  every figure has a table equivalent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg

__all__ = [
    "CATEGORICAL",
    "use_project_style",
    "plot_loss_trajectory",
    "plot_schedules",
    "plot_sgd_convergence",
    "plot_alpha_curve",
    "plot_regularization_path",
    "plot_correlation_heatmap",
    "save",
]

# Validated categorical order (light surface). Do not reorder or cycle.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

# Single-hue ordinal ramp for ordered magnitudes (light -> dark).
BLUE_RAMP: tuple[str, ...] = ("#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")

# Diverging pair with a neutral midpoint, for correlations.
DIVERGING = ("#184f95", "#f0efec", "#d03b3b")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"


def use_project_style() -> None:
    """Apply the project's matplotlib defaults. Call once per script."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "figure.dpi": 120,
            "font.size": 10,
        }
    )


def _label_line_end(ax, x, y, text: str, color: str, dx: float = 1.01) -> None:
    """Direct-label a line at its right end, in ink rather than the series hue."""
    if len(x) == 0:
        return
    ax.annotate(
        text,
        xy=(x[-1], y[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8.5,
        color=INK_SOFT,
    )


def save(fig, name: str, out_dir: Path = cfg.FIGURES_DIR) -> Path:
    """Write a figure to ``figures/<name>.png`` and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_loss_trajectory(steps: Sequence[tuple[str, float]]):
    """Waterfall of the four validation-loss adjustments."""
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(range(len(values)), values, color=CATEGORICAL[0], marker="o", markersize=6)
    for i, v in enumerate(values):
        ax.annotate(
            f"{v:.4f}",
            (i, v),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
            color=INK_SOFT,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("validation loss (MSE)")
    ax.set_title("A +5% then -5% round trip does not return to the start")
    ax.margins(y=0.22)
    fig.tight_layout()
    return fig


def plot_schedules(histories: Mapping[str, Sequence[float]]):
    """Compare learning-rate schedules over epochs."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for slot, (name, values) in enumerate(histories.items()):
        color = CATEGORICAL[slot % len(CATEGORICAL)]
        x = np.arange(len(values))
        y = np.asarray(values, dtype=float)
        ax.plot(x, y, color=color, label=name)
        # Cosine annealing ends at exactly 0, which a log axis cannot draw --
        # label the last value that is actually on the scale.
        positive = np.flatnonzero(y > 0)
        if len(positive):
            cut = positive[-1] + 1
            _label_line_end(ax, x[:cut], y[:cut], name, color)

    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.set_title("Three ways to shrink a learning rate")
    ax.legend(loc="lower left")
    ax.margins(x=0.12)
    fig.tight_layout()
    return fig


def plot_sgd_convergence(curve: pd.DataFrame, divergence_headroom: float = 4.0):
    """Training RMSE per epoch at several constant learning rates.

    A diverged run can be twelve orders of magnitude above the others, which would
    flatten every informative curve into a horizontal line. So the y-axis is
    scaled to the runs that stayed finite, and each diverged run is labelled where
    it leaves the top of the plot rather than being allowed to set the scale.
    """
    rates = sorted(curve["learning_rate"].unique())
    finite_all = curve[np.isfinite(curve["train_rmse"])]
    baseline = float(finite_all["train_rmse"].min())

    # Ceiling from the runs that behave, not from the one that exploded.
    well_behaved = [
        float(curve[curve["learning_rate"] == lr]["train_rmse"].max())
        for lr in rates
        if np.isfinite(curve[curve["learning_rate"] == lr]["train_rmse"]).all()
        and float(curve[curve["learning_rate"] == lr]["train_rmse"].max())
        < baseline * 1e3
    ]
    ceiling = (max(well_behaved) if well_behaved else baseline) * divergence_headroom

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for i, lr in enumerate(rates):
        sub = curve[curve["learning_rate"] == lr]
        color = BLUE_RAMP[min(i, len(BLUE_RAMP) - 1)]
        finite = sub[np.isfinite(sub["train_rmse"])]
        ax.plot(finite["epoch"], finite["train_rmse"], color=color, label=f"{lr:g}")

        exceeded = finite[finite["train_rmse"] > ceiling]
        if len(exceeded):
            peak = float(finite["train_rmse"].max())
            ax.annotate(
                f"eta0 = {lr:g} diverged\n(RMSE {peak:.1e}, off scale)",
                xy=(float(exceeded["epoch"].iloc[0]), ceiling),
                xytext=(8, -26),
                textcoords="offset points",
                fontsize=8.5,
                color="#d03b3b",
            )
        elif len(finite):
            _label_line_end(
                ax,
                finite["epoch"].to_numpy(),
                finite["train_rmse"].to_numpy(),
                f"{lr:g}",
                color,
            )

    ax.set_ylim(baseline * 0.75, ceiling)
    ax.set_xlim(1, float(curve["epoch"].max()) * 1.1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training RMSE (percentage points)")
    ax.set_yscale("log")
    ax.set_title("Learning rate decides whether training converges at all")
    ax.legend(
        title="constant learning rate (eta0)",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=5,
    )
    fig.tight_layout()
    return fig


def plot_alpha_curve(tuned: pd.DataFrame):
    """Cross-validated RMSE against regularization strength."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(tuned["alpha"], tuned["rmse_mean"], color=CATEGORICAL[0])
    ax.fill_between(
        tuned["alpha"],
        tuned["rmse_mean"] - tuned["rmse_sd"],
        tuned["rmse_mean"] + tuned["rmse_sd"],
        color=CATEGORICAL[0],
        alpha=0.14,
        linewidth=0,
    )

    best = tuned.attrs.get("best_alpha")
    one_se = tuned.attrs.get("one_se_alpha")
    for value, label, color in ((best, "best", CATEGORICAL[1]), (one_se, "1-SE rule", CATEGORICAL[6])):
        if value is None:
            continue
        ax.axvline(value, color=color, linestyle="--", linewidth=1.4)
        ax.annotate(
            f"{label}\nalpha = {value:.3g}",
            xy=(value, ax.get_ylim()[1]),
            xytext=(4, -14),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SOFT,
            va="top",
        )

    ax.set_xscale("log")
    ax.set_xlabel("alpha (regularization strength, log scale)")
    ax.set_ylabel("CV RMSE, state-grouped folds")
    ax.set_title("Too little penalty overfits; too much underfits")
    fig.tight_layout()
    return fig


def plot_regularization_path(path: pd.DataFrame, top_n_labels: int = 4):
    """Standardized coefficients as the penalty tightens."""
    feature_cols = [c for c in path.columns if c != "alpha"]
    final_strength = path[feature_cols].iloc[0].abs().sort_values(ascending=False)
    to_label = set(final_strength.head(top_n_labels).index)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    endpoints: list[tuple[float, str]] = []
    for slot, col in enumerate(feature_cols):
        color = CATEGORICAL[slot % len(CATEGORICAL)]
        label = cfg.MEASURE_LABELS.get(col, col)
        ax.plot(path["alpha"], path[col], color=color, label=label)
        if col in to_label:
            endpoints.append((float(path[col].iloc[-1]), label))

    # Nudge direct labels apart so converging lines do not overprint each other.
    y_span = float(np.ptp(ax.get_ylim()))
    min_gap = y_span * 0.075
    placed: list[float] = []
    for y_value, label in sorted(endpoints):
        y_text = y_value
        for taken in placed:
            if abs(y_text - taken) < min_gap:
                y_text = taken + min_gap
        placed.append(y_text)
        ax.annotate(
            label,
            xy=(path["alpha"].iloc[-1], y_value),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=INK_SOFT,
        ).set_position((7, (y_text - y_value) / y_span * ax.bbox.height))

    ax.axhline(0, color=INK_SOFT, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (regularization strength, log scale)")
    ax.set_ylabel("coefficient (standardized predictors)")
    model = path.attrs.get("model", "ridge")
    ax.set_title(f"{model.title()} path: collinear predictors stop fighting as alpha rises")
    ax.set_xmargin(0.0)
    ax.set_xlim(path["alpha"].min(), path["alpha"].max() * 40)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=3)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame):
    """Diverging heatmap of predictor correlations, with values printed."""
    cmap = mpl.colors.LinearSegmentedColormap.from_list("divergent", DIVERGING)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.grid(False)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.to_numpy()[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#ffffff" if abs(value) > 0.78 else INK,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson correlation", color=INK_SOFT)
    cbar.outline.set_visible(False)
    ax.set_title("Social-need measures move together -- hence the penalty")
    fig.tight_layout()
    return fig
