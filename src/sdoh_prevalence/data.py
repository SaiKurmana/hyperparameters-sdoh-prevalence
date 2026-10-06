"""Get county-level prevalence + social-need data into a tidy analytic frame.

Three entry points, one interface:

* :func:`download_places` -- pull the real CDC PLACES county file (no API key).
* :func:`make_synthetic`  -- generate correlated fake data so the repo runs and
  the tests pass with no network access.
* :func:`load_dataset`    -- the loader every script uses; reads PLACES, your own
  CSV, or synthetic data and always returns the same wide frame.

The wide frame is one row per county: the outcome column, one column per
predictor, plus geography and population columns.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .config import ColumnMap

log = logging.getLogger(__name__)

__all__ = [
    "available_measures",
    "download_places",
    "long_to_wide",
    "make_synthetic",
    "load_dataset",
    "describe_coverage",
]


# --------------------------------------------------------------------------- #
# Live download
# --------------------------------------------------------------------------- #


def _soql_in(values: Iterable[str]) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"({quoted})"


def available_measures(timeout: int = 60) -> pd.DataFrame:
    """Return the measure IDs the live PLACES county dataset actually contains.

    Called before every download so a renamed or retired measure produces a
    readable error instead of an empty file. CDC re-publishes PLACES annually and
    measure IDs are not guaranteed stable across releases.
    """
    import requests

    params = {
        "$select": "measureid, categoryid, short_question_text",
        "$group": "measureid, categoryid, short_question_text",
        "$order": "measureid",
        "$limit": "500",
    }
    resp = requests.get(cfg.PLACES_JSON_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def download_places(
    measures: Sequence[str] | None = None,
    data_value_type: str = cfg.DATA_VALUE_TYPE_ID,
    out_path: Path = cfg.PLACES_RAW_CSV,
    validate: bool = True,
    timeout: int = 120,
) -> Path:
    """Download the requested PLACES county measures to ``out_path`` as CSV.

    Pages through the SODA API until a short page comes back. Writes the long
    format (one row per county x measure) unchanged, so the raw file stays a
    faithful copy of what CDC served.
    """
    import requests

    measures = list(measures or (cfg.OUTCOME_MEASURE, *cfg.SDOH_MEASURES))

    if validate:
        live = available_measures(timeout=timeout)
        known = set(live["measureid"])
        missing = [m for m in measures if m not in known]
        if missing:
            raise ValueError(
                f"These measure IDs are not in the live PLACES county dataset: "
                f"{missing}.\nThe release may have renamed or retired them. "
                f"Available IDs: {sorted(known)}\n"
                f"Check {cfg.PLACES_LANDING_PAGE} and update "
                f"SDOH_MEASURES / OUTCOME_MEASURE in config.py."
            )

    where = (
        f"measureid in {_soql_in(measures)} "
        f"and datavaluetypeid = '{data_value_type}'"
    )

    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        params = {
            "$select": ", ".join(cfg.PLACES_FIELDS),
            "$where": where,
            "$order": "locationid, measureid",
            "$limit": str(cfg.PAGE_SIZE),
            "$offset": str(offset),
        }
        resp = requests.get(cfg.PLACES_BASE_URL, params=params, timeout=timeout)
        resp.raise_for_status()

        page = pd.read_csv(io.StringIO(resp.text), dtype={"locationid": str})
        log.info("fetched %d rows at offset %d", len(page), offset)
        if page.empty:
            break
        frames.append(page)
        if len(page) < cfg.PAGE_SIZE:
            break
        offset += cfg.PAGE_SIZE

    if not frames:
        raise RuntimeError(
            "PLACES returned no rows. Check the measure IDs and "
            f"data_value_type ({data_value_type!r})."
        )

    raw = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_path, index=False)
    log.info("wrote %d rows to %s", len(raw), out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Reshaping
# --------------------------------------------------------------------------- #


def long_to_wide(
    long_df: pd.DataFrame,
    outcome: str = cfg.OUTCOME_MEASURE,
    predictors: Sequence[str] = cfg.SDOH_MEASURES,
) -> pd.DataFrame:
    """Pivot the PLACES long format to one row per county.

    Counties missing any requested measure are dropped, and how many were
    dropped is logged -- silent row loss is how an analysis quietly stops
    representing the population it claims to.
    """
    df = long_df.copy()
    df["locationid"] = df["locationid"].astype(str).str.zfill(5)

    wanted = [outcome, *predictors]
    df = df[df["measureid"].isin(wanted)]

    wide = df.pivot_table(
        index=["locationid", "locationname", "stateabbr"],
        columns="measureid",
        values="data_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    pop = (
        df.groupby("locationid")[["totalpopulation", "totalpop18plus"]]
        .first()
        .reset_index()
    )
    wide = wide.merge(pop, on="locationid", how="left")

    before = len(wide)
    wide = wide.dropna(subset=wanted)
    dropped = before - len(wide)
    if dropped:
        log.warning(
            "dropped %d of %d counties (%.1f%%) missing at least one measure",
            dropped,
            before,
            100 * dropped / before,
        )

    return wide.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #


def make_synthetic(
    n_counties: int = 3_100,
    predictors: Sequence[str] = cfg.SDOH_MEASURES,
    noise_sd: float = 1.2,
    random_state: int = cfg.RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a correlated, PLACES-shaped dataset with known coefficients.

    Deliberately built so the regularization lesson has something to bite on:
    the social-need predictors share a single latent "deprivation" factor, so
    they are strongly collinear, and only three of them carry real signal.

    The true coefficients are attached as ``df.attrs["true_coefficients"]``.
    """
    rng = np.random.default_rng(random_state)
    n_pred = len(predictors)

    # One latent deprivation factor drives every social-need measure, which is
    # roughly how these variables behave in the real data.
    deprivation = rng.normal(0.0, 1.0, size=n_counties)
    loadings = rng.uniform(0.55, 0.9, size=n_pred)
    idiosyncratic = rng.normal(0.0, 1.0, size=(n_counties, n_pred))

    z = deprivation[:, None] * loadings + idiosyncratic * np.sqrt(1 - loadings**2)

    # Put each predictor on a plausible prevalence scale (percent of adults).
    means = rng.uniform(8.0, 26.0, size=n_pred)
    sds = rng.uniform(2.5, 6.0, size=n_pred)
    x = np.clip(means + z * sds, 0.5, 85.0)

    # Only a few predictors truly matter; the rest are collinear passengers.
    true_coefs = np.zeros(n_pred)
    signal_idx = rng.choice(n_pred, size=min(3, n_pred), replace=False)
    true_coefs[signal_idx] = rng.uniform(0.10, 0.28, size=len(signal_idx))

    y = 4.2 + (x - means) @ true_coefs + rng.normal(0.0, noise_sd, size=n_counties)
    y = np.clip(y, 2.0, 30.0)

    state_pool = np.array(
        ["AL", "AR", "AZ", "CA", "GA", "IA", "KY", "LA", "MS", "NC",
         "NY", "OH", "OK", "SC", "TN", "TX", "VA", "WA", "WV", "WI"]
    )
    states = rng.choice(state_pool, size=n_counties)

    df = pd.DataFrame(x, columns=list(predictors))
    df.insert(0, "stateabbr", states)
    df.insert(0, "locationname", [f"Synthetic County {i:04d}" for i in range(n_counties)])
    df.insert(0, "locationid", [f"{90_000 + i:05d}" for i in range(n_counties)])
    df[cfg.OUTCOME_MEASURE] = y
    df["totalpopulation"] = rng.lognormal(10.3, 1.1, size=n_counties).round()
    df["totalpop18plus"] = (df["totalpopulation"] * rng.uniform(0.72, 0.80, n_counties)).round()

    df.attrs["true_coefficients"] = dict(zip(predictors, true_coefs.round(4), strict=True))
    df.attrs["synthetic"] = True
    return df


# --------------------------------------------------------------------------- #
# The one loader the scripts call
# --------------------------------------------------------------------------- #


def load_dataset(
    path: Path | str | None = None,
    columns: ColumnMap = cfg.DEFAULT_COLUMNS,
    synthetic: bool = False,
    allow_synthetic_fallback: bool = True,
) -> pd.DataFrame:
    """Return the wide analytic frame from whichever source is available.

    Resolution order:

    1. ``synthetic=True``            -> generated data.
    2. ``path``                      -> your own CSV (long or wide; mapped via
       ``columns``).
    3. the downloaded PLACES raw CSV -> pivoted to wide.
    4. synthetic, with a warning, if ``allow_synthetic_fallback``.

    Step 4 exists so a fresh clone runs before anyone has downloaded anything.
    Every frame carries ``df.attrs["source"]`` so a script can print, and a
    reader can see, which of the four it actually got.
    """
    if synthetic:
        df = make_synthetic(predictors=columns.predictors)
        df.attrs["source"] = "synthetic"
        return df

    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No dataset at {path}")
        raw = pd.read_csv(path, dtype={columns.geo_id: str})
        df = _coerce_to_wide(raw, columns)
        df.attrs["source"] = str(path)
        return df

    if cfg.PLACES_RAW_CSV.exists():
        raw = pd.read_csv(cfg.PLACES_RAW_CSV, dtype={"locationid": str})
        df = long_to_wide(raw, columns.outcome, columns.predictors)
        df.attrs["source"] = str(cfg.PLACES_RAW_CSV)
        return df

    if not allow_synthetic_fallback:
        raise FileNotFoundError(
            f"{cfg.PLACES_RAW_CSV} not found. Run: python -m scripts.download_data"
        )

    log.warning(
        "No downloaded data found at %s -- falling back to SYNTHETIC data. "
        "Run `python -m scripts.download_data` for the real CDC PLACES file.",
        cfg.PLACES_RAW_CSV,
    )
    df = make_synthetic(predictors=columns.predictors)
    df.attrs["source"] = "synthetic (fallback)"
    return df


def _coerce_to_wide(raw: pd.DataFrame, columns: ColumnMap) -> pd.DataFrame:
    """Accept either PLACES long format or an already-wide file."""
    if {"measureid", "data_value"}.issubset(raw.columns):
        return long_to_wide(raw, columns.outcome, columns.predictors)

    missing = [c for c in columns.required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Found: {list(raw.columns)}\n"
            "Adjust ColumnMap(outcome=..., predictors=(...)) to match your file."
        )
    keep = [*columns.required, *columns.extra_keep]
    if columns.weight and columns.weight in raw.columns:
        keep.append(columns.weight)
    return raw[list(dict.fromkeys(keep))].dropna(
        subset=[columns.outcome, *columns.predictors]
    ).reset_index(drop=True)


def describe_coverage(df: pd.DataFrame, columns: ColumnMap = cfg.DEFAULT_COLUMNS) -> pd.DataFrame:
    """One row per variable: n, mean, sd, min, max -- a quick sanity table."""
    cols = [columns.outcome, *columns.predictors]
    out = df[cols].agg(["count", "mean", "std", "min", "max"]).T
    out.index = [cfg.MEASURE_LABELS.get(i, i) for i in out.index]
    return out.round(2)
