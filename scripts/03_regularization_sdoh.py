"""Regularization strength on collinear social-need predictors.

Run top to bottom, or step through the ``# %%`` cells in VS Code.

This is the hyperparameter that matters most for this kind of population-health
model. County-level social-need measures are strongly correlated with each other —
counties with high food insecurity also tend to have high housing insecurity, more
food stamp receipt, and less reliable transportation. An unpenalized regression
handles that badly: it hands out large, unstable, mutually cancelling coefficients
that change sign if you refit on a different sample.
"""

# %%
import sys
from pathlib import Path

SRC = (
    Path(__file__).resolve().parents[1] / "src"
    if "__file__" in globals()
    else Path.cwd() / "src"
)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sdoh_prevalence import config as cfg  # noqa: E402
from sdoh_prevalence import plots  # noqa: E402
from sdoh_prevalence.data import describe_coverage, load_dataset  # noqa: E402
from sdoh_prevalence.hyperparameters import alpha_grid  # noqa: E402
from sdoh_prevalence.models import (  # noqa: E402
    compare_models,
    correlation_table,
    cross_validated_rmse,
    make_scarce_regime,
    make_xy,
    regularization_path,
    tune_alpha,
)

plots.use_project_style()
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

# %% [markdown]
# # 1. The data
#
# Outcome: **diagnosed diabetes prevalence** among adults, crude, by county.
# Predictors: the **health-related social needs** measures plus uninsured share.
# Source: CDC PLACES county data (model-based small-area estimates from BRFSS).
#
# `load_dataset()` reads the downloaded PLACES file if present, your own CSV if you
# pass a path, and otherwise falls back to synthetic data with a warning — so this
# script runs on a fresh clone.

# %%
df = load_dataset()
print(f"source:   {df.attrs.get('source')}")
print(f"counties: {len(df):,}")
print(f"states:   {df['stateabbr'].nunique()}")
print()
print(describe_coverage(df))

# %%
data = make_xy(df)
corr = correlation_table(data)
print(corr)

off_diagonal = corr.to_numpy()[~np.eye(len(corr), dtype=bool)]
print()
print(f"mean |correlation| between predictors: {np.abs(off_diagonal).mean():.2f}")
print(f"max  |correlation| between predictors: {np.abs(off_diagonal).max():.2f}")

# %%
fig = plots.plot_correlation_heatmap(corr)
print("saved:", plots.save(fig, "03_correlation"))

# %% [markdown]
# # 2. Why a penalty, in one table
#
# `alpha` is the regularization strength: how much the fit is charged for large
# coefficients. Ridge (L2) shrinks every coefficient toward zero; lasso (L1) shrinks
# some to *exactly* zero, which selects variables.
#
# > **Analogy:** collinear predictors are co-authors who each claim credit for the
# > same paper. Unpenalized, they argue loudly — one takes +8, another −6, and the
# > sum happens to fit the training counties. Ridge makes credit expensive, so they
# > settle on a shared, modest, stable story. Lasso instead picks one author and
# > drops the rest — efficient, but *which* one it picks is close to arbitrary among
# > near-identical cousins, and that choice is not evidence about causes.
#
# Note the cross-validation: folds are grouped by **state**, not random. Counties in
# the same state share BRFSS sampling design, Medicaid expansion status, and state
# policy. A random split scatters near-siblings across both sides and reports a
# score that will not survive contact with a new state.

# %%
comparison = compare_models(data)
print(comparison.to_string(index=False))

# %% [markdown]
# Read the `coef_l2_norm` and `n_nonzero_coefs` columns alongside `cv_rmse`. The
# penalized fits usually give up a trivial amount of in-sample R² and buy back
# stability — smaller coefficients that mean something you can report.

# %% [markdown]
# # 3. Tuning alpha
#
# Always search penalty strength on a **log** grid. The gap between `alpha=0.01` and
# `alpha=0.1` matters enormously; the gap between `100` and `100.09` does not. A
# linear grid spends nearly its whole budget where nothing changes.

# %%
alphas = alpha_grid(1e-3, 1e3, n=45)
ridge_tuned = tune_alpha(data, alphas=alphas, model="ridge")

print(f"counties: {len(data):,}   predictors: {data.X.shape[1]}")
print(f"best alpha:        {ridge_tuned.attrs['best_alpha']:.4g}")
print(f"best CV RMSE:      {ridge_tuned.attrs['best_rmse']:.4f}")
print(f"1-SE rule alpha:   {ridge_tuned.attrs['one_se_alpha']:.4g}")
print()
print(ridge_tuned.iloc[::6].round(4).to_string(index=False))

spread = ridge_tuned["rmse_mean"].max() - ridge_tuned["rmse_mean"].min()
print(f"\nCV RMSE spread across five orders of magnitude of alpha: {spread:.4f}")

# %% [markdown]
# ## 3a. An honest result worth pausing on
#
# On the full file the curve is nearly **flat** until alpha gets very large. With
# ~3,100 counties and 8 predictors there is far too much data to overfit 8
# coefficients, so the penalty has almost nothing to do. Tuning alpha here is close
# to a waste of compute.
#
# That is the situation most regularization tutorials quietly demonstrate on, and it
# teaches the wrong instinct. Regularization earns its keep when the number of
# parameters approaches the number of observations — which in applied
# population-health work happens constantly:
#
# - you restrict to **one state's counties** (n ≈ 80), or to rural counties only;
# - you add **interaction and squared terms** because you think social needs
#   compound rather than simply add;
# - you move from counties to **census tracts** with a much richer feature set;
# - you have 40 candidate predictors and 200 units, which is an ordinary SDOH file.
#
# `make_scarce_regime()` builds the first two: 120 counties, and 8 predictors
# expanded to 44 at degree 2. Now alpha decides the model.

# %%
scarce = make_scarce_regime(df, n_counties=120, degree=2)
print(f"counties: {len(scarce)}   predictors: {scarce.X.shape[1]}")

scarce_alphas = alpha_grid(1e-2, 1e4, n=40)
scarce_tuned = tune_alpha(scarce, alphas=scarce_alphas, model="ridge", n_splits=4)

print(f"best alpha:      {scarce_tuned.attrs['best_alpha']:.4g}")
print(f"best CV RMSE:    {scarce_tuned.attrs['best_rmse']:.4f}")
print(f"1-SE rule alpha: {scarce_tuned.attrs['one_se_alpha']:.4g}")
print()
print(scarce_tuned.iloc[::5].round(3).to_string(index=False))

worst = scarce_tuned["rmse_mean"].max()
best = scarce_tuned["rmse_mean"].min()
print(f"\nbest vs worst alpha: {best:.3f} vs {worst:.3f} RMSE "
      f"({100 * (worst - best) / best:.0f}% worse)")

# %%
fig = plots.plot_alpha_curve(scarce_tuned)
print("saved:", plots.save(fig, "03_alpha_curve"))

# %% [markdown]
# *That* curve is U-shaped, and both walls are real failure modes:
#
# - **Left wall (alpha too small)** — the model overfits; held-out error rises.
# - **Right wall (alpha too large)** — every coefficient is crushed toward zero and
#   the model underfits, predicting close to the national mean for every county.
#
# The **1-SE rule** picks the largest alpha whose CV error is still within one
# standard error of the best. It deliberately trades a sliver of accuracy for a
# simpler, more stable model — usually the right call when the coefficients will be
# read by humans making programmatic decisions, not just scored.
#
# The practical lesson from the two regimes together: **check whether a
# hyperparameter matters before spending a search budget on it.** One flat curve tells
# you to go work on the features, the sample, or the question instead. A hyperparameter
# search that cannot change the answer is expensive theatre.

# %% [markdown]
# # 4. The regularization path
#
# The path shows every coefficient as alpha rises. This is the clearest picture of
# collinearity you can produce.

# %%
ridge_path = regularization_path(data, alphas=alphas, model="ridge")
lasso_path = regularization_path(data, alphas=alphas, model="lasso")

feature_cols = [c for c in ridge_path.columns if c != "alpha"]
print("Ridge coefficients (standardized) at three strengths:\n")
show = ridge_path.set_index("alpha")[feature_cols]
print(show.iloc[[0, len(show) // 2, -1]].round(3).T)

# %%
fig = plots.plot_regularization_path(ridge_path)
print("saved:", plots.save(fig, "03_ridge_path"))

fig = plots.plot_regularization_path(lasso_path)
print("saved:", plots.save(fig, "03_lasso_path"))

# %%
nonzero = (lasso_path[feature_cols].abs() > 1e-8).sum(axis=1)
survival = pd.DataFrame({"alpha": lasso_path["alpha"], "n_nonzero": nonzero})
print("Lasso: how many predictors survive as the penalty tightens\n")
print(survival.iloc[::6].to_string(index=False))

# %% [markdown]
# # 5. Final model, and the caveats that outrank every hyperparameter
#
# Fit at the tuned alpha and report the coefficients on the standardized scale, so
# they are comparable: each is the change in diabetes prevalence (percentage points)
# per one-standard-deviation increase in that social-need measure.

# %%
from sklearn.linear_model import Ridge  # noqa: E402

from sdoh_prevalence.models import build_pipeline  # noqa: E402

best_alpha = ridge_tuned.attrs["best_alpha"]
final = build_pipeline(Ridge(alpha=best_alpha)).fit(data.X, data.y)
coefs = pd.Series(
    final.named_steps["model"].coef_, index=data.feature_names
).sort_values(key=abs, ascending=False)
coefs.index = [cfg.MEASURE_LABELS.get(i, i) for i in coefs.index]

mean_rmse, sd_rmse = cross_validated_rmse(Ridge(alpha=best_alpha), data)
print(f"Ridge(alpha={best_alpha:.4g})")
print(f"state-grouped CV RMSE: {mean_rmse:.3f} +/- {sd_rmse:.3f} percentage points\n")
print("standardized coefficients (pp per 1 SD):")
print(coefs.round(3).to_string())

if df.attrs.get("synthetic"):
    print("\nSynthetic data -- true coefficients used to generate it:")
    print(pd.Series(df.attrs["true_coefficients"]).round(3).to_string())

# %% [markdown]
# ## What no amount of tuning fixes
#
# Worth stating plainly, because a well-tuned model invites over-reading:
#
# 1. **Ecological fallacy.** These are county-level associations. They do not license
#    any claim about an individual's diabetes risk given their food security.
# 2. **PLACES estimates are modeled, not measured.** Both the outcome and the
#    social-need predictors come from small-area estimation over BRFSS, using
#    demographic and geographic inputs. Predictors and outcome therefore share parts
#    of a data-generating model, which inflates apparent fit. A low RMSE here is
#    partly the two estimates agreeing with each other rather than with the world.
# 3. **Not causal.** Nothing here identifies an effect. It is a description of
#    covariation, useful for targeting where to look, not for claiming what to change.
# 4. **Crude, not age-adjusted.** Set `DATA_VALUE_TYPE_ID = "AgeAdjPrv"` in
#    `config.py` if the question is a fair between-county comparison rather than
#    present burden. That is a study-design choice, and it will move the
#    coefficients more than any alpha you pick.
#
# Hyperparameter tuning improves how well a model fits the data you gave it. It has
# nothing to say about whether that data can answer your question.
