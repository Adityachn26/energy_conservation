"""Phases 2–4: loading, quality checks/cleaning and hourly aggregation."""

import pandas as pd
import pytest

from src.data.aggregation import aggregate_to_hourly, hourly_counts, select_complete_hours
from src.data.loader import DataValidationError, load_raw_csv
from src.data.preprocessing import clean_raw_data, data_quality_report
from tests.conftest import full_hour, make_raw


def test_loader_renames_and_types_columns(tmp_path, synthetic_raw_original):
    path = tmp_path / "raw.csv"
    synthetic_raw_original.head(500).to_csv(path, index=False)
    df = load_raw_csv(path)
    assert list(df.columns) == ["meter", "timestamp", "kwh", "voltage", "current", "frequency"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert len(df) == 500


def test_loader_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"time": [1], "energy": [2]}).to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        load_raw_csv(path)


def test_loader_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_sample_data"):
        load_raw_csv(tmp_path / "nope.csv")


def test_cleaning_removes_and_counts_bad_rows():
    rows = full_hour("A", "2020-01-01 00:00")
    rows += [rows[0]]  # exact duplicate
    rows += [("A", "2020-01-01 00:03", 0.5)]  # duplicate (meter, timestamp), different value
    rows += [("A", "2020-01-01 01:00", -1.0)]  # negative kWh
    df = make_raw(rows)

    report = data_quality_report(df)
    assert report["duplicate_rows"] == 1
    assert report["negative_kwh"] == 1

    clean, removed = clean_raw_data(df)
    assert removed == {
        "exact_duplicates": 1,
        "duplicate_meter_timestamps": 1,
        "missing_kwh": 0,
        "negative_kwh": 1,
        "total_removed": 3,
    }
    assert len(clean) == 20


def test_hourly_aggregation_sums_kwh_and_averages_electrical_values():
    df = make_raw(full_hour("A", "2020-01-01 00:00", kwh_per_reading=0.01))
    df.loc[:9, "voltage"] = 220.0  # half the readings at 220 V, half at 230 V
    hourly = aggregate_to_hourly(df)
    row = hourly.iloc[0]
    assert row["kwh"] == pytest.approx(0.20)
    assert row["voltage"] == pytest.approx(225.0)
    assert row["readings_per_hour"] == 20 and row["is_complete"]


def test_empty_hours_are_missing_not_zero():
    # Hours 00 and 03 present; 01 and 02 have no readings at all.
    df = make_raw(full_hour("A", "2020-01-01 00:00") + full_hour("A", "2020-01-01 03:00"))
    hourly = aggregate_to_hourly(df)
    assert len(hourly) == 4  # grid includes the empty hours explicitly
    empty = hourly.loc[hourly["readings_per_hour"] == 0]
    assert len(empty) == 2
    assert empty["kwh"].isna().all(), "empty hours must be NaN, never 0 kWh"
    assert hourly_counts(hourly) == {"hourly_rows": 4, "complete_hours": 2, "empty_hours": 2, "partial_hours": 0}
    assert len(select_complete_hours(hourly)) == 2


def test_partial_hours_are_not_treated_as_complete():
    df = make_raw(full_hour("A", "2020-01-01 00:00")[:15])  # 15 of 20 readings
    hourly = aggregate_to_hourly(df)
    assert not hourly.iloc[0]["is_complete"]
    assert pd.isna(hourly.iloc[0]["kwh"])  # under-counted sum would be misleading


def test_grid_is_built_per_meter_span():
    df = make_raw(
        full_hour("A", "2020-01-01 00:00") + full_hour("A", "2020-01-01 02:00") + full_hour("B", "2020-01-01 05:00")
    )
    hourly = aggregate_to_hourly(df)
    assert hourly.groupby("meter").size().to_dict() == {"A": 3, "B": 1}
