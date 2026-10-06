"""Hyperparameters, taught on county-level chronic disease and social-need data.

Typical use from a script:

    from sdoh_prevalence import config as cfg
    from sdoh_prevalence.data import load_dataset
    from sdoh_prevalence.models import make_xy, tune_alpha

    df = load_dataset()
    data = make_xy(df)
    results = tune_alpha(data)
"""

from __future__ import annotations

__version__ = "0.2.0"

from . import config, data, hyperparameters, models, plots  # noqa: F401

__all__ = ["config", "data", "hyperparameters", "models", "plots", "__version__"]
