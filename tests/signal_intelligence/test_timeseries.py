"""Tests for signal_intelligence.timeseries (Phase 6)."""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from signal_intelligence.timeseries import (
    _quarter_end,
    align_with_earnings,
    build_mention_matrix,
    build_timeseries,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(tmp_path: Path, records: list[dict]) -> Path:
    """Write a minimal JSONL PTC pool and return its path."""
    from ptc_extraction.io import append_to_pool
    from ptc_extraction.schema import PTC

    pool_path = tmp_path / "pool.jsonl"
    ptcs = []
    for r in records:
        ptcs.append(
            PTC(
                mechanism=r.get("mechanism", "A mechanism long enough for validation"),
                raw_text=r.get("raw_text", "raw text here"),
                direction=r.get("direction", "precursor"),
                polarity=r.get("polarity", -1),
                source_date=r["source_date"],
                source_type=r.get("source_type", "management"),
                source_id=r["source_id"],
                extracted_by="test",
                extraction_version="v1.1",
            )
        )
    append_to_pool(pool_path, ptcs)
    return pool_path


def _make_canonical_df(source_id_groups: dict[str, list[str]]) -> pd.DataFrame:
    """Build a minimal canonical DataFrame with canonical_id → source_ids mapping."""
    rows = []
    for cid, sids in source_id_groups.items():
        rows.append(
            {
                "canonical_id": cid,
                "ticker": "ML.PA",
                "corpus": "management",
                "direction": "precursor",
                "polarity": -1,
                "corpus_presence": "insider_only",
                "sector_tag": "untagged",
                "n_ptcs": len(sids),
                "is_noise": False,
                "representative_mechanism": "mechanism text",
                "source_ids": json.dumps(sids),
                "matched_canonical_id": None,
                "cross_map_similarity": None,
            }
        )
    return pd.DataFrame(rows)


def _make_earnings_df(dates_surprises: list[tuple]) -> pd.DataFrame:
    idx = pd.Index([d for d, _ in dates_surprises], name="date")
    return pd.DataFrame(
        {"surprise_pct": [s for _, s in dates_surprises]},
        index=idx,
    )


# ---------------------------------------------------------------------------
# _quarter_end
# ---------------------------------------------------------------------------


def test_quarter_end_q1():
    ts = pd.Timestamp("2024-02-15")
    assert _quarter_end(ts) == pd.Timestamp("2024-03-31")


def test_quarter_end_q4():
    ts = pd.Timestamp("2024-10-01")
    assert _quarter_end(ts) == pd.Timestamp("2024-12-31")


# ---------------------------------------------------------------------------
# build_mention_matrix
# ---------------------------------------------------------------------------


def test_build_mention_matrix_basic(tmp_path):
    pool_path = _make_pool(
        tmp_path,
        [
            {"source_id": "ML.PA-management-2024-01-15-doc1", "source_date": date(2024, 1, 15)},
            {"source_id": "ML.PA-management-2024-02-20-doc2", "source_date": date(2024, 2, 20)},
            {"source_id": "ML.PA-management-2024-07-10-doc3", "source_date": date(2024, 7, 10)},
        ],
    )
    canonical_df = _make_canonical_df(
        {
            "c1": ["ML.PA-management-2024-01-15-doc1", "ML.PA-management-2024-02-20-doc2"],
            "c2": ["ML.PA-management-2024-07-10-doc3"],
        }
    )
    matrix = build_mention_matrix(canonical_df, pool_path)
    assert "c1" in matrix.columns
    assert "c2" in matrix.columns
    # Both Jan and Feb are in Q1 2024
    q1_end = pd.Timestamp("2024-03-31")
    assert matrix.at[q1_end, "c1"] == 2
    q3_end = pd.Timestamp("2024-09-30")
    assert matrix.at[q3_end, "c2"] == 1


def test_build_mention_matrix_empty_pool(tmp_path):
    pool_path = tmp_path / "empty.jsonl"
    pool_path.write_text("")
    canonical_df = _make_canonical_df({"c1": ["some-source-id"]})
    matrix = build_mention_matrix(canonical_df, pool_path)
    assert matrix.empty


def test_build_mention_matrix_unknown_source_id(tmp_path):
    pool_path = _make_pool(
        tmp_path,
        [{"source_id": "ML.PA-management-2024-01-15-doc1", "source_date": date(2024, 1, 15)}],
    )
    canonical_df = _make_canonical_df({"c1": ["not-in-pool"]})
    matrix = build_mention_matrix(canonical_df, pool_path)
    assert matrix.empty


# ---------------------------------------------------------------------------
# align_with_earnings
# ---------------------------------------------------------------------------


def test_align_with_earnings_basic(tmp_path):
    pool_path = _make_pool(
        tmp_path,
        [{"source_id": "ML.PA-management-2024-01-15-doc1", "source_date": date(2024, 1, 15)}],
    )
    canonical_df = _make_canonical_df({"c1": ["ML.PA-management-2024-01-15-doc1"]})
    matrix = build_mention_matrix(canonical_df, pool_path)

    # Earnings announced in Q1 2024 quarter
    earnings_df = _make_earnings_df([(date(2024, 2, 28), 5.2)])
    aligned = align_with_earnings(matrix, earnings_df)

    assert len(aligned) == 1
    assert aligned.iloc[0]["canonical_id"] == "c1"
    assert aligned.iloc[0]["mention_count"] == 1
    assert abs(aligned.iloc[0]["surprise_pct"] - 5.2) < 0.01


def test_align_with_earnings_no_overlap(tmp_path):
    pool_path = _make_pool(
        tmp_path,
        [{"source_id": "ML.PA-management-2024-01-15-doc1", "source_date": date(2024, 1, 15)}],
    )
    canonical_df = _make_canonical_df({"c1": ["ML.PA-management-2024-01-15-doc1"]})
    matrix = build_mention_matrix(canonical_df, pool_path)

    # Earnings only in 2023 — no overlap with 2024 mention matrix
    earnings_df = _make_earnings_df([(date(2023, 11, 15), 2.0)])
    aligned = align_with_earnings(matrix, earnings_df)
    assert aligned.empty


def test_align_with_earnings_empty_matrix():
    earnings_df = _make_earnings_df([(date(2024, 2, 28), 5.2)])
    aligned = align_with_earnings(pd.DataFrame(), earnings_df)
    assert aligned.empty


# ---------------------------------------------------------------------------
# build_timeseries (integration)
# ---------------------------------------------------------------------------


def test_build_timeseries_saves_files(tmp_path):
    pool_path = _make_pool(
        tmp_path,
        [{"source_id": "ML.PA-management-2024-01-15-doc1", "source_date": date(2024, 1, 15)}],
    )
    canonical_df = _make_canonical_df({"c1": ["ML.PA-management-2024-01-15-doc1"]})
    earnings_df = _make_earnings_df([(date(2024, 2, 28), 5.2)])
    output_dir = tmp_path / "processed"

    matrix, aligned = build_timeseries("ML.PA", canonical_df, pool_path, earnings_df, output_dir)

    assert (output_dir / "ML.PA_mention_matrix.csv").exists()
    assert (output_dir / "ML.PA_aligned.csv").exists()
    assert not matrix.empty
    assert not aligned.empty
