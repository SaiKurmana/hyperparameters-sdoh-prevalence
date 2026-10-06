"""Data layer: synthetic generation, long->wide reshaping, and the loader."""

import numpy as np
import pandas as pd
import pytest

from sdoh_prevalence import config as cfg
from sdoh_prevalence.config import ColumnMap
from sdoh_prevalence.data import (
    describe_coverage,
    load_dataset,
    long_to_wide,
    make_synthetic,
)


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic(n_counties=400)


def test_synthetic_shape_and_columns(synthetic):
    expected = {
        "locationid",
        "locationname",
        "stateabbr",
        cfg.OUTCOME_MEASURE,
        "totalpopulation",
        "totalpop18plus",
        *cfg.SDOH_MEASURES,
    }
    assert expected.issubset(synthetic.columns)
    assert len(synthetic) == 400
    assert synthetic["locationid"].is_unique


def test_synthetic_values_are_plausible_prevalences(synthetic):
    for col in (cfg.OUTCOME_MEASURE, *cfg.SDOH_MEASURES):
        assert synthetic[col].between(0, 100).all(), col
    assert synthetic[cfg.OUTCOME_MEASURE].std() > 0


def test_synthetic_predictors_are_collinear_by_design(synthetic):
    """The regularization lesson needs correlated predictors to be worth teaching."""
    corr = synthetic[list(cfg.SDOH_MEASURES)].corr().to_numpy()
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    assert np.abs(off_diagonal).mean() > 0.25


def test_synthetic_is_reproducible():
    a = make_synthetic(n_counties=50, random_state=7)
    b = make_synthetic(n_counties=50, random_state=7)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_records_true_coefficients(synthetic):
    truth = synthetic.attrs["true_coefficients"]
    assert set(truth) == set(cfg.SDOH_MEASURES)
    assert sum(abs(v) > 0 for v in truth.values()) == 3


def test_long_to_wide_pivots_and_drops_incomplete_counties():
    long_df = pd.DataFrame(
        [
            # county 01001 has both measures; 01003 is missing the outcome
            ("01001", "Complete", "AL", "DIABETES", 12.0, 1000, 800),
            ("01001", "Complete", "AL", "FOODINSECU", 20.0, 1000, 800),
            ("01003", "Partial", "AL", "FOODINSECU", 22.0, 2000, 1500),
        ],
        columns=[
            "locationid",
            "locationname",
            "stateabbr",
            "measureid",
            "data_value",
            "totalpopulation",
            "totalpop18plus",
        ],
    )
    wide = long_to_wide(long_df, outcome="DIABETES", predictors=("FOODINSECU",))

    assert len(wide) == 1
    assert wide.loc[0, "locationid"] == "01001"
    assert wide.loc[0, "DIABETES"] == 12.0
    assert wide.loc[0, "FOODINSECU"] == 20.0


def test_long_to_wide_zero_pads_fips():
    long_df = pd.DataFrame(
        [
            (1001, "Padded", "AL", "DIABETES", 12.0, 1000, 800),
            (1001, "Padded", "AL", "FOODINSECU", 20.0, 1000, 800),
        ],
        columns=[
            "locationid",
            "locationname",
            "stateabbr",
            "measureid",
            "data_value",
            "totalpopulation",
            "totalpop18plus",
        ],
    )
    wide = long_to_wide(long_df, outcome="DIABETES", predictors=("FOODINSECU",))
    assert wide.loc[0, "locationid"] == "01001"


def test_load_dataset_synthetic_flag_is_labelled():
    df = load_dataset(synthetic=True)
    assert df.attrs["source"] == "synthetic"
    assert len(df) > 100


def test_load_dataset_reads_own_wide_csv(tmp_path):
    df = make_synthetic(n_counties=60)
    path = tmp_path / "mine.csv"
    df.to_csv(path, index=False)

    loaded = load_dataset(path=path)
    assert loaded.attrs["source"] == str(path)
    assert len(loaded) == 60


def test_load_dataset_reports_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"locationid": ["01001"], "something_else": [1.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path=path)


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(path=tmp_path / "nope.csv")


def test_custom_column_map_round_trips(tmp_path):
    raw = pd.DataFrame(
        {
            "fips": ["01001", "01003", "01005"],
            "county": ["A", "B", "C"],
            "st": ["AL", "AL", "MS"],
            "dm_prev": [12.0, 13.5, 11.0],
            "food_insec": [20.0, 24.0, 18.0],
            "no_transport": [8.0, 9.5, 7.0],
        }
    )
    path = tmp_path / "custom.csv"
    raw.to_csv(path, index=False)

    columns = ColumnMap(
        outcome="dm_prev",
        predictors=("food_insec", "no_transport"),
        geo_id="fips",
        geo_name="county",
        group="st",
        weight=None,
    )
    loaded = load_dataset(path=path, columns=columns)
    assert len(loaded) == 3
    assert {"dm_prev", "food_insec", "no_transport"}.issubset(loaded.columns)


def test_describe_coverage_shape(synthetic):
    table = describe_coverage(synthetic)
    assert list(table.columns) == ["count", "mean", "std", "min", "max"]
    assert len(table) == 1 + len(cfg.SDOH_MEASURES)
