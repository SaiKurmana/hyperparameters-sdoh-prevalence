"""Hyperparameters and assignment operators, in population-health terms.

Run this file top to bottom (``python scripts/01_assignment_operators.py``) or
step through it cell by cell in VS Code: click "Run Cell" above any ``# %%``
marker, or put the cursor in a cell and press Shift+Enter. The Interactive Window
keeps variables between cells exactly like a notebook, but the file stays a plain
``.py`` that git can diff and pytest can import.

Scenario for the whole script: you are modeling **county-level diagnosed diabetes
prevalence** from **health-related social needs** (food insecurity, housing
insecurity, lack of transportation, and so on) using CDC PLACES data. Every
hyperparameter below is a knob on that model.
"""

# %% [markdown]
# # 1. Hyperparameters vs. parameters
#
# **Hyperparameters** are the settings you choose *before* training. **Parameters**
# are what the model learns *from the data during* training.
#
# In this project:
#
# | | Example | Who sets it |
# | --- | --- | --- |
# | Hyperparameter | ridge penalty `alpha`, learning rate, number of epochs, CV fold count | you, before fitting |
# | Parameter | the coefficient on food insecurity | the fitting algorithm |
#
# > **Analogy for an epidemiologist:** hyperparameters are your *study design* —
# > case definition, sampling frame, matching ratio. Parameters are the *estimates*
# > the study produces. You cannot fix a bad design by staring harder at the odds
# > ratio, and you cannot fix a badly tuned model by staring at its coefficients.
#
# Common hyperparameters and what they control:
#
# - **Learning rate** — how far the weights move per update. Too large and the fit
#   diverges; too small and it never arrives.
# - **Batch size** — how many counties per gradient update.
# - **Number of epochs** — how many passes over the training counties.
# - **Regularization strength (`alpha`)** — how hard the model is penalized for
#   large coefficients. The single most consequential knob in this project, because
#   social-need predictors are heavily collinear.
# - **Momentum** — how much of the previous update carries into the next.

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

from sdoh_prevalence import plots  # noqa: E402
from sdoh_prevalence.hyperparameters import (  # noqa: E402
    as_scientific,
    simulate_loss_adjustments,
)

plots.use_project_style()
print("ready")

# %% [markdown]
# # 2. Exercise: adjusting model values with assignment operators
#
# You are tracking the model's learning rate and its validation loss (mean squared
# error on held-out counties) as you work. Each cell modifies a value using a
# **compound assignment operator**.
#
# | Operator | Means | Use it for |
# | --- | --- | --- |
# | `+=` | `x = x + n` | changes in the metric's own units |
# | `-=` | `x = x - n` | changes in the metric's own units |
# | `*=` | `x = x * n` | changes stated as percentages |
#
# The rule worth memorizing: **absolute change → `+=` / `-=`; relative change →
# `*=`**. "Decrease by 10%" is `*= 0.9`, never `-= 0.10`. Confusing the two is the
# same error as subtracting 10 from a rate when you meant to reduce it by a tenth.

# %%
# The initial learning rate for the model
learning_rate = 0.01

# The initial training loss of the model
training_loss = 0.8

print(f"learning_rate = {learning_rate}")
print(f"training_loss = {training_loss}")

# %% [markdown]
# ## 2a. Decrease the learning rate by 10% for better stability
#
# Context: the first epochs on the county data overshot — the loss bounced instead
# of descending. Shrinking the step size steadies it.

# %%
# Use the multiplication assignment operator to decrease
# the learning_rate variable by 10% to account for better stability
learning_rate *= 0.9

print(learning_rate)

# %%
### Notebook grading
if learning_rate == 0.01 * 0.9:
    print("Nice job! You can use the multiplication assignment operator like this: learning_rate *= 0.9")
elif learning_rate == 0.01 * 1.1:
    print("Sorry! learning_rate is too high. It looks like you increased the learning rate by 10% instead of decreasing it. Try again")
else:
    print("Sorry! learning_rate is incorrect. Try again and make sure to use the *= assignment operator")

# %% [markdown]
# ## 2b. Add a small absolute amount to the loss for noisy data
#
# Context: PLACES estimates for small-population counties carry wide confidence
# intervals. Including them adds measurement noise the model cannot explain, which
# raises the floor on achievable loss. That is an *absolute* addition in MSE units.

# %%
# Use the addition assignment operator to add
# a small value (0.02) to the training_loss variable due to noisy data
training_loss += 0.02

print(training_loss)

# %%
### Notebook grading
if training_loss == 0.8 + 0.02:
    print("Nice job! You can use the addition assignment operator like this: training_loss += 0.02")
elif training_loss == 0.8:
    print("Sorry! training_loss is too low. Did you add the noise? Try again")
else:
    print("Sorry! training_loss is incorrect. Try again and make sure to use the += assignment operator")

# %% [markdown]
# ## 2c. Increase the loss by 5% to account for overfitting
#
# Context: with eight collinear social-need predictors and no penalty, the model
# starts fitting quirks of individual counties. Held-out loss rises by a
# *proportion* of where it already was.

# %%
# Use the multiplication assignment operator to increase
# training_loss by 5% to account for overfitting
training_loss *= 1.05

print(training_loss)

# %%
### Notebook grading
if training_loss == (0.8 + 0.02) * 1.05:
    print("Nice job! You can use the multiplication assignment operator like this: training_loss *= 1.05")
elif training_loss == 0.8 + 0.02:
    print("Sorry! training_loss is too low. Did you account for overfitting? Try again")
else:
    print("Sorry! training_loss is incorrect. Try again and make sure to use the *= assignment operator")

# %% [markdown]
# ## 2d. Decrease the loss by 5% to account for regularization
#
# Context: an L2 penalty shrinks those competing coefficients and the model
# generalizes better. Script 03 does this for real on the actual data.

# %%
# Use the multiplication assignment operator to decrease
# training_loss by 5% to account for regularization
training_loss *= 0.95

print(training_loss)

# %%
### Notebook grading
if training_loss == (0.8 + 0.02) * 1.05 * 0.95:
    print("Nice job! You can use the multiplication assignment operator like this: training_loss *= 0.95")
elif training_loss == (0.8 + 0.02) * 1.05:
    print("Sorry! training_loss is too high. Did you account for regularization? Try again")
else:
    print("Sorry! training_loss is incorrect. Try again and make sure to use the *= assignment operator")

# %% [markdown]
# ## 2e. Subtract a small absolute amount for improved generalization
#
# Context: extending coverage from one state's counties to all ~3,140 US counties
# gives the model far more to learn from.

# %%
# Use the subtraction assignment operator to subtract
# a small value (0.05) from training_loss to account
# for improved generalization due to more training data.
training_loss -= 0.05

print(training_loss)

# %%
### Notebook grading
if training_loss == (0.8 + 0.02) * 1.05 * 0.95 - 0.05:
    print("Nice job! You can use the subtraction assignment operator like this: training_loss -= 0.05")
elif training_loss == (0.8 + 0.02) * 1.05 * 0.95:
    print("Sorry! training_loss is too high. Did you account for the improved generalization? Try again")
else:
    print("Sorry! training_loss is incorrect. Try again and make sure to use the -= assignment operator")

# %% [markdown]
# ## 2f. The trajectory, and why +5% then −5% is not a round trip
#
# `0.82 × 1.05 × 0.95 = 0.81795`, not `0.82`. Relative changes compound, so a 5%
# rise followed by a 5% fall is a small net **decrease** — the second percentage
# applies to a larger base than the first did.
#
# This is the same arithmetic that makes "incidence rose 20% then fell 20%" a net
# decline, and it is a routine source of error in reporting rate changes.

# %%
steps = simulate_loss_adjustments()

previous = None
for label, value in steps:
    delta = "" if previous is None else f"  ({value - previous:+.5f})"
    print(f"{label:<32} {value:.5f}{delta}")
    previous = value

print()
print(f"net change from start:   {steps[-1][1] - steps[0][1]:+.5f}")
print(f"+5% then -5% on 0.82 ->  {0.82 * 1.05 * 0.95:.5f}   (not 0.82)")

# %%
fig = plots.plot_loss_trajectory(steps)
print("saved:", plots.save(fig, "01_loss_trajectory"))

# %% [markdown]
# ## 2g. A caveat on the graders above
#
# Those graders compare floats with `==`, which is safe here *only* because both
# sides perform identical arithmetic in identical order. In general, binary
# floating point does not give you the number you wrote down.

# %%
import math  # noqa: E402

print(f"0.1 + 0.2 == 0.3        -> {0.1 + 0.2 == 0.3}")
print(f"repr(0.1 + 0.2)         -> {0.1 + 0.2!r}")
print(f"math.isclose(...)       -> {math.isclose(0.1 + 0.2, 0.3)}")
print()
print("In real code, compare with a tolerance: math.isclose() or pytest.approx().")
print("See tests/test_hyperparameters.py.")

# %% [markdown]
# # 3. Learning rates and scientific notation
#
# Learning rates are small, so they are written in scientific notation: `1e-3` is
# Python for $1 \times 10^{-3}$, i.e. `0.001`. It is only a literal — same `float`,
# fewer zeros to miscount. Miscounting zeros is exactly how a learning rate ends up
# 10× too large.

# %%
# Initial learning rate
learning_rate = 1e-3  # Equivalent to 0.001

# Update the learning rate by decreasing it by 10%
learning_rate *= 0.9

print(f"Updated learning rate: {learning_rate}")
print(f"In scientific notation: {as_scientific(learning_rate)}")

# %%
print(f"1e-3 == 0.001 -> {1e-3 == 0.001}")
print(f"type(1e-3)    -> {float}")
print()
for exponent in range(1, 7):
    lr = 10**-exponent
    print(f"1e-{exponent} = {lr:<12} reads as {lr:.0e}")

# %% [markdown]
# # Takeaways
#
# 1. Hyperparameters are the study design; parameters are the estimates.
# 2. Absolute change → `+=` / `-=`. Relative change → `*=`. "Down 10%" is `*= 0.9`.
# 3. Relative changes compound: `*= 1.05` then `*= 0.95` is a net decrease.
# 4. `1e-3` is just `0.001`, written so you cannot miscount zeros.
# 5. Never compare floats with `==` outside a controlled exercise.
#
# Next: `02_learning_rate_schedules.py` turns one `*= 0.9` into a schedule, and
# shows what learning rate does to real convergence.
