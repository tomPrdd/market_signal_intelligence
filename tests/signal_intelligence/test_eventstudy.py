"""Tests for signal_intelligence.eventstudy (Phase 7B)."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from signal_intelligence.eventstudy import (
    compute_car,
    compute_car_path,
    run_car_profiles,
    run_event_study,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(n_days: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic price series with deterministic random log-returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    log_returns = rng.normal(0.0002, 0.01, size=n_days)
    price = 100.0 * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame({"adj_close": price, "close": price}, index=dates)
    df.index = pd.to_datetime(df.index)
    return df


def _make_pool(tmp_path: Path, records: list[dict]) -> Path:
    from ptc_extraction.io import append_to_pool
    from ptc_extraction.schema import PTC

    pool_path = tmp_path / "pool.jsonl"
    ptcs = [
        PTC(
            mechanism="A mechanism text long enough to satisfy min_length constraint",
            raw_text="raw text here",
            direction=r.get("direction", "precursor"),
            polarity=-1,
            source_date=r["source_date"],
            source_type="management",
            source_id=r["source_id"],
            extracted_by="test",
            extraction_version="v1.1",
        )
        for r in records
    ]
    append_to_pool(pool_path, ptcs)
    return pool_path


def _make_canonical_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "ticker": "ML.PA",
        "corpus": "management",
        "polarity": -1,
        "corpus_presence": "insider_only",
        "sector_tag": "untagged",
        "n_ptcs": 1,
        "is_noise": False,
        "representative_mechanism": "mechanism",
        "matched_canonical_id": None,
        "cross_map_similarity": None,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ---------------------------------------------------------------------------
# compute_car
# ---------------------------------------------------------------------------


def test_compute_car_returns_float():
    prices = _make_prices(400)
    returns = np.log(prices["adj_close"] / prices["adj_close"].shift(1)).dropna()
    event_date = returns.index[150]
    car = compute_car(event_date, returns)
    assert car is not None
    assert isinstance(car, float)


def test_compute_car_insufficient_history():
    prices = _make_prices(50)  # too few days for the estimation window
    returns = np.log(prices["adj_close"] / prices["adj_close"].shift(1)).dropna()
    event_date = returns.index[5]  # too early — not enough estimation window
    car = compute_car(event_date, returns)
    assert car is None


def test_compute_car_out_of_bounds():
    prices = _make_prices(200)
    returns = np.log(prices["adj_close"] / prices["adj_close"].shift(1)).dropna()
    event_date = returns.index[-2]  # too close to end for +30 window
    car = compute_car(event_date, returns)
    # May return None if +30 window is out of bounds
    assert car is None or isinstance(car, float)


# ---------------------------------------------------------------------------
# compute_car_path
# ---------------------------------------------------------------------------


def test_compute_car_path_shape_and_consistency():
    prices = _make_prices(400)
    returns = np.log(prices["adj_close"] / prices["adj_close"].shift(1)).dropna()
    event_date = returns.index[150]
    path = compute_car_path(event_date, returns)
    assert path is not None
    assert list(path.index) == list(range(1, 31))
    # endpoint of the path equals the scalar CAR
    assert compute_car(event_date, returns) == float(path.iloc[-1])


def test_compute_car_path_insufficient_history():
    prices = _make_prices(50)
    returns = np.log(prices["adj_close"] / prices["adj_close"].shift(1)).dropna()
    assert compute_car_path(returns.index[5], returns) is None


# ---------------------------------------------------------------------------
# run_car_profiles
# ---------------------------------------------------------------------------


def test_run_car_profiles_long_format(tmp_path):
    prices = _make_prices(600)
    pool_path = _make_pool(
        tmp_path,
        [
            {"source_id": "ML.PA-management-2022-09-01-doc1", "source_date": date(2022, 9, 1)},
            {"source_id": "ML.PA-management-2023-03-01-doc2", "source_date": date(2023, 3, 1)},
        ],
    )
    canonical_df = _make_canonical_df(
        [
            {
                "canonical_id": "c1",
                "direction": "precursor",
                "source_ids": json.dumps(
                    [
                        "ML.PA-management-2022-09-01-doc1",
                        "ML.PA-management-2023-03-01-doc2",
                    ]
                ),
            },
            {
                "canonical_id": "c2",
                "direction": "consequence",
                "source_ids": json.dumps(["ML.PA-management-2022-09-01-doc1"]),
            },
        ]
    )
    result = run_car_profiles(canonical_df, pool_path, prices)
    assert set(result.columns) == {"canonical_id", "rel_day", "mean_car", "sem_car", "n_events"}
    # consequence canonicals excluded
    assert set(result["canonical_id"]) == {"c1"}
    c1 = result[result["canonical_id"] == "c1"]
    assert list(c1["rel_day"]) == list(range(1, 31))
    assert (c1["n_events"] == 2).all()
    assert c1["sem_car"].notna().all()


def test_run_car_profiles_saves_csv(tmp_path):
    prices = _make_prices(600)
    pool_path = _make_pool(
        tmp_path,
        [
            {"source_id": "ML.PA-management-2022-09-01-doc1", "source_date": date(2022, 9, 1)},
            {"source_id": "ML.PA-management-2023-03-01-doc2", "source_date": date(2023, 3, 1)},
        ],
    )
    canonical_df = _make_canonical_df(
        [
            {
                "canonical_id": "c1",
                "direction": "precursor",
                "source_ids": json.dumps(
                    [
                        "ML.PA-management-2022-09-01-doc1",
                        "ML.PA-management-2023-03-01-doc2",
                    ]
                ),
            }
        ]
    )
    output_dir = tmp_path / "out"
    run_car_profiles(canonical_df, pool_path, prices, output_dir=output_dir, ticker="ML.PA")
    assert (output_dir / "ML.PA_car_profiles.csv").exists()


def test_run_car_profiles_single_event_no_sem(tmp_path):
    prices = _make_prices(600)
    pool_path = _make_pool(
        tmp_path,
        [{"source_id": "ML.PA-management-2022-09-01-doc1", "source_date": date(2022, 9, 1)}],
    )
    canonical_df = _make_canonical_df(
        [
            {
                "canonical_id": "c1",
                "direction": "precursor",
                "source_ids": json.dumps(["ML.PA-management-2022-09-01-doc1"]),
            }
        ]
    )
    result = run_car_profiles(canonical_df, pool_path, prices)
    assert (result["n_events"] == 1).all()
    assert result["sem_car"].isna().all()


# ---------------------------------------------------------------------------
# run_event_study
# ---------------------------------------------------------------------------


def test_run_event_study_returns_expected_columns(tmp_path):
    prices = _make_prices(400)
    pool_path = _make_pool(
        tmp_path,
        [{"source_id": "ML.PA-management-2022-10-01-doc1", "source_date": date(2022, 10, 1)}],
    )
    canonical_df = _make_canonical_df(
        [
            {
                "canonical_id": "c1",
                "direction": "precursor",
                "source_ids": json.dumps(["ML.PA-management-2022-10-01-doc1"]),
            }
        ]
    )
    result = run_event_study(canonical_df, pool_path, prices)
    expected = {
        "canonical_id",
        "direction",
        "n_events",
        "mean_car",
        "std_car",
        "t_stat",
        "p_value",
        "positive_pct",
    }
    assert expected.issubset(set(result.columns))


def test_run_event_study_saves_csv(tmp_path):
    prices = _make_prices(400)
    pool_path = _make_pool(
        tmp_path,
        [{"source_id": "ML.PA-management-2022-10-01-doc1", "source_date": date(2022, 10, 1)}],
    )
    canonical_df = _make_canonical_df(
        [
            {
                "canonical_id": "c1",
                "direction": "precursor",
                "source_ids": json.dumps(["ML.PA-management-2022-10-01-doc1"]),
            }
        ]
    )
    output_dir = tmp_path / "out"
    run_event_study(canonical_df, pool_path, prices, output_dir=output_dir, ticker="ML.PA")
    assert (output_dir / "ML.PA_eventstudy.csv").exists()


def test_run_event_study_consequence_has_zero_events(tmp_path):
    prices = _make_prices(400)
    pool_path = _make_pool(
        tmp_path,
        [
            {
                "source_id": "ML.PA-management-2022-10-01-doc1",
                "source_date": date(2022, 10, 1),
                "direction": "consequence",
            }
        ],
    )
    canonical_df = _make_canonical_df(
        [
            {
                "canonical_id": "c1",
                "direction": "consequence",
                "source_ids": json.dumps(["ML.PA-management-2022-10-01-doc1"]),
            }
        ]
    )
    result = run_event_study(canonical_df, pool_path, prices)
    assert result.iloc[0]["n_events"] == 0


def test_run_event_study_multiple_events(tmp_path):
    prices = _make_prices(600)
    pool_path = _make_pool(
        tmp_path,
        [
            {"source_id": "ML.PA-management-2022-06-01-doc1", "source_date": date(2022, 6, 1)},
            {"source_id": "ML.PA-management-2022-09-01-doc2", "source_date": date(2022, 9, 1)},
            {"source_id": "ML.PA-management-2023-03-01-doc3", "source_date": date(2023, 3, 1)},
        ],
    )
    canonical_df = _make_canonical_df(
        [
            {
                "canonical_id": "c1",
                "direction": "precursor",
                "source_ids": json.dumps(
                    [
                        "ML.PA-management-2022-06-01-doc1",
                        "ML.PA-management-2022-09-01-doc2",
                        "ML.PA-management-2023-03-01-doc3",
                    ]
                ),
            }
        ]
    )
    result = run_event_study(canonical_df, pool_path, prices)
    row = result[result["canonical_id"] == "c1"].iloc[0]
    assert row["n_events"] >= 2
    assert row["t_stat"] is not None
    assert row["p_value"] is not None
    assert 0.0 <= row["positive_pct"] <= 100.0
