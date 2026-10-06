"""Learning rate: from one `*= 0.9` to a schedule, then to real convergence.

Run top to bottom, or step through the ``# %%`` cells in VS Code.

Script 01 shrank a learning rate once. Here the same operation is applied every
epoch, which is what a *schedule* is, and then we watch what different learning
rates actually do to a model fitted on county data by stochastic gradient descent.
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

import pandas as pd  # noqa: E402

from sdoh_prevalence import plots  # noqa: E402
from sdoh_prevalence.data import load_dataset  # noqa: E402
from sdoh_prevalence.hyperparameters import (  # noqa: E402
    cosine_decay,
    exponential_decay,
    schedule_history,
    step_decay,
)
from sdoh_prevalence.models import make_xy, sgd_learning_curve  # noqa: E402

plots.use_project_style()
pd.set_option("display.width", 120)

# %% [markdown]
# # 1. Repeated multiplication is exponential decay
#
# `lr *= 0.9` once takes 10% off. Applied every epoch for 20 epochs it does **not**
# reach zero — it reaches 12% of where it started. Repeated multiplication is
# exponential, and exponential decay has a long tail.
#
# > **Analogy:** walking downhill in fog. Long strides while the slope is obviously
# > downward, shorter strides as the ground flattens, so you don't stride past the
# > lowest point and back up the other side. A schedule is the plan for shortening
# > your stride — set before you start walking, not decided at each step.

# %%
initial_lr = 1e-3
epochs = 30

schedules = {
    "exponential (10%/epoch)": schedule_history(
        lambda e: exponential_decay(initial_lr, e, decay=0.10), epochs
    ),
    "step (halve every 10)": schedule_history(
        lambda e: step_decay(initial_lr, e, drop=0.5, every=10), epochs
    ),
    "cosine annealing": schedule_history(
        lambda e: cosine_decay(initial_lr, e, total_epochs=epochs), epochs
    ),
}

table = pd.DataFrame(schedules)
table.index.name = "epoch"
print(table.iloc[[0, 5, 10, 15, 20, 25, 30]].map(lambda v: f"{v:.3e}"))
print()
print("share of the initial rate remaining at epoch 20:")
for name, values in schedules.items():
    print(f"  {name:<26} {values[20] / initial_lr:6.1%}")

# %%
fig = plots.plot_schedules(schedules)
print("saved:", plots.save(fig, "02_schedules"))

# %% [markdown]
# Why the three differ in practice:
#
# - **Exponential** decays smoothly and never quite stops shrinking. Safe default.
# - **Step** holds a rate steady, then drops it. The plateaus make runs easy to
#   compare, and the drops often line up with visible jumps in validation loss.
# - **Cosine** starts slow, spends most of its budget mid-range, and lands at zero
#   exactly when the epoch budget runs out — useful when you have committed to a
#   fixed number of epochs.
#
# Note that the schedule itself has hyperparameters: the decay rate, the drop
# factor, the interval. Tuning a scheduler is tuning hyperparameters about
# hyperparameters, which is why fixing a sensible schedule early and spending your
# search budget on the penalty (script 03) is usually the better trade.

# %% [markdown]
# # 2. What learning rate actually does to convergence
#
# Now the same idea against real data. We fit the diabetes-prevalence model by
# stochastic gradient descent at five constant learning rates and record training
# RMSE after every epoch.
#
# If no data has been downloaded yet the loader falls back to synthetic data and
# says so — the shape of the result is the same either way. Run
# `python scripts/download_data.py` for the real CDC PLACES file.

# %%
df = load_dataset()
print(f"source: {df.attrs.get('source')}")
print(f"counties: {len(df):,}")

data = make_xy(df)
curve = sgd_learning_curve(data, learning_rates=(1e-6, 1e-5, 1e-3, 1e-1, 1.0), epochs=60)

summary = (
    curve.groupby("learning_rate")
    .agg(
        epochs_survived=("epoch", "max"),
        final_rmse=("train_rmse", "last"),
        best_rmse=("train_rmse", "min"),
    )
    .round(4)
)
print(summary)

# %%
fig = plots.plot_sgd_convergence(curve)
print("saved:", plots.save(fig, "02_sgd_convergence"))

# %% [markdown]
# # 3. Reading the result
#
# The pattern is consistent, and it is the reason learning rate is usually the
# first hyperparameter anyone tunes:
#
# - **Too small** (`1e-6`, `1e-5`): the curve is still falling when the epochs run
#   out, and it never reaches the floor the other rates find. Nothing looks broken —
#   which is exactly what makes this the expensive failure. You conclude the social
#   need measures carry little signal, when in fact you just stopped early.
# - **About right** (`1e-3`): fast descent to a flat floor, which here is the noise
#   the data genuinely cannot explain.
# - **Slightly too large** (`1e-1`): it converges to a *worse* floor, bouncing around
#   the minimum instead of settling into it.
# - **Far too large** (`1.0`): the loss explodes — twelve orders of magnitude above
#   where it started, on its way to `inf`. The run fails loudly, which is the *cheap*
#   failure, because you cannot miss it.
#
# A practical rule: find the largest rate that does not diverge, then back off by
# roughly an order of magnitude and add a decay schedule.
#
# An epidemiological caution on what we just did: RMSE here is **training** error on
# county-level PLACES estimates. It measures convergence of the optimizer, not the
# validity of the model. Script 03 introduces held-out error with state-grouped
# folds, which is the number that would actually tell you whether the model
# transfers to a state it has never seen.
#
# Next: `03_regularization_sdoh.py`.
