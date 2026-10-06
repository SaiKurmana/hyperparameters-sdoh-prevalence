"""Project configuration: paths, CDC PLACES identifiers, and column mappings.

Everything here is a plain constant or dataclass so it can be overridden from a
script without touching the library code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

PLACES_RAW_CSV = RAW_DIR / "places_county_long.csv"
ANALYTIC_CSV = PROCESSED_DIR / "county_analytic.csv"

# --------------------------------------------------------------------------- #
# CDC PLACES (Socrata / SODA API, no API key required)
# --------------------------------------------------------------------------- #

# "PLACES: Local Data for Better Health, County Data" -- long format, one row
# per county x measure x data-value-type.
PLACES_DATASET_ID = "swc5-untb"
PLACES_BASE_URL = f"https://data.cdc.gov/resource/{PLACES_DATASET_ID}.csv"
PLACES_JSON_URL = f"https://data.cdc.gov/resource/{PLACES_DATASET_ID}.json"

# CDC republishes PLACES annually and has, in the past, changed both dataset IDs
# and measure IDs between releases. Every download therefore validates the
# requested measures against the live measure list before pulling rows -- see
# data.available_measures().
PLACES_LANDING_PAGE = (
    "https://data.cdc.gov/500-Cities-Places/"
    "PLACES-Local-Data-for-Better-Health-County-Data-20/swc5-untb"
)

# Crude prevalence, not age-adjusted. Crude is the right choice when the model's
# job is to describe the burden a county's health department actually faces;
# switch to "AgeAdjPrv" when comparing counties with very different age
# structures, which is a modeling decision, not a hyperparameter.
DATA_VALUE_TYPE_ID = "CrdPrv"

# Outcome: diagnosed diabetes prevalence among adults.
OUTCOME_MEASURE = "DIABETES"

# Health-related social needs measures (PLACES' SDOH block, sourced from the
# BRFSS social determinants module) plus two access measures. These are
# deliberately correlated with each other -- that is the point of the
# regularization exercise.
SDOH_MEASURES: tuple[str, ...] = (
    "FOODINSECU",   # food insecurity, past 12 months
    "FOODSTAMP",    # received food stamps, past 12 months
    "HOUSINSECU",   # housing insecurity, past 12 months
    "SHUTUTILITY",  # utility services shutoff threat, past 12 months
    "LACKTRPT",     # lack of reliable transportation, past 12 months
    "ISOLATION",    # feelings of loneliness
    "EMOTIONSPT",   # lack of social and emotional support
    "ACCESS2",      # no health insurance, adults 18-64
)

#: Human-readable labels for plots and tables.
MEASURE_LABELS: dict[str, str] = {
    "DIABETES": "Diagnosed diabetes",
    "FOODINSECU": "Food insecurity",
    "FOODSTAMP": "Food stamp receipt",
    "HOUSINSECU": "Housing insecurity",
    "SHUTUTILITY": "Utility shutoff threat",
    "LACKTRPT": "No reliable transportation",
    "ISOLATION": "Loneliness",
    "EMOTIONSPT": "No social/emotional support",
    "ACCESS2": "Uninsured (18-64)",
}

# Fields pulled from the API. Names are the Socrata API field names, which are
# lowercased versions of the CSV headers.
PLACES_FIELDS: tuple[str, ...] = (
    "year",
    "stateabbr",
    "statedesc",
    "locationname",
    "locationid",
    "categoryid",
    "measureid",
    "datavaluetypeid",
    "data_value",
    "low_confidence_limit",
    "high_confidence_limit",
    "totalpopulation",
    "totalpop18plus",
)

PAGE_SIZE = 50_000  # Socrata caps a single page; the loader pages until empty.


# --------------------------------------------------------------------------- #
# Bring-your-own-data mapping
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ColumnMap:
    """How to read a dataset that is *not* CDC PLACES.

    Point ``outcome`` and ``predictors`` at the columns in your own file and the
    rest of the pipeline works unchanged.
    """

    outcome: str = OUTCOME_MEASURE
    predictors: tuple[str, ...] = SDOH_MEASURES
    geo_id: str = "locationid"
    geo_name: str = "locationname"
    group: str = "stateabbr"
    weight: str | None = "totalpop18plus"
    extra_keep: tuple[str, ...] = field(default_factory=tuple)

    @property
    def required(self) -> tuple[str, ...]:
        cols = (self.outcome, *self.predictors, self.geo_id, self.geo_name, self.group)
        return tuple(c for c in cols if c)


DEFAULT_COLUMNS = ColumnMap()

RANDOM_STATE = 20260913
