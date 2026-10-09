"""Tests for signal_intelligence.pipeline (orchestrator).

All external calls (Bedrock, yfinance, FMP, Exa) are mocked.
Tests verify that phases are called in the right order, results are wired
correctly, and phase toggles are respected.
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from signal_intelligence.pipeline import PipelineConfig, PipelineResult, run_pipeline

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_pool(tmp_path: Path) -> Path:
    from ptc_extraction.io import append_to_pool
    from ptc_extraction.schema import PTC

    pool_path = tmp_path / "pool.jsonl"
    ptcs = [
        PTC(
            mechanism="Rising raw material costs compress margins significantly over time",
            raw_text="raw text here",
            direction="precursor",
            polarity=-1,
            source_date=date(2024, 1, 15),
            source_type="management",
            source_id="ML.PA-management-2024-01-15-doc1",
            extracted_by="test",
            extraction_version="v1.1",
        )
    ]
    append_to_pool(pool_path, ptcs)
    return pool_path


def _make_fake_embeddings(tmp_path: Path, ticker: str) -> None:
    interim = tmp_path / "interim"
    interim.mkdir(exist_ok=True)
    vecs = np.random.rand(1, 1024).astype(np.float32)
    np.save(interim / f"{ticker}_embeddings.npy", vecs)
    pd.DataFrame(
        [
            {
                "content_hash": "abc123",
                "source_id": f"{ticker}-management-2024-01-15-doc1",
                "source_type": "management",
                "direction": "precursor",
                "polarity": -1,
                "source_date": "2024-01-15",
                "mechanism": "Rising raw material costs compress margins significantly",
            }
        ]
    ).to_csv(interim / f"{ticker}_embedding_index.csv", index=False)


def _make_fake_canonical(tmp_path: Path, ticker: str) -> pd.DataFrame:
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "canonical_id": f"{ticker}-management-0",
                "ticker": ticker,
                "corpus": "management",
                "direction": "precursor",
                "polarity": -1,
                "corpus_presence": "insider_only",
                "sector_tag": "untagged",
                "n_ptcs": 1,
                "is_noise": False,
                "representative_mechanism": "Rising costs compress margins",
                "source_ids": json.dumps([f"{ticker}-management-2024-01-15-doc1"]),
                "matched_canonical_id": None,
                "cross_map_similarity": None,
            }
        ]
    )
    df.to_csv(processed / f"{ticker}_canonicals.csv", index=False)
    return df


def _fake_prices(n: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(0)
    price = 100.0 * np.exp(rng.normal(0, 0.01, n).cumsum())
    df = pd.DataFrame({"adj_close": price, "close": price}, index=pd.to_datetime(dates))
    df.index.name = "date"
    return df


def _fake_earnings() -> pd.DataFrame:
    idx = pd.Index([date(2024, 2, 28)], name="date")
    return pd.DataFrame(
        {"reported_eps": [1.5], "estimated_eps": [1.3], "surprise_pct": [15.38]},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_result_has_expected_attributes():
    result = PipelineResult(ticker="ML.PA")
    assert result.canonical_df.empty
    assert result.granger_df.empty
    assert result.eventstudy_df.empty


def test_pipeline_skips_all_phases_when_disabled(tmp_path):
    pool_path = _make_pool(tmp_path)
    cfg = PipelineConfig(
        ticker="ML.PA",
        pool_path=pool_path,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        run_embedding=False,
        run_clustering=False,
        run_timeseries=False,
        run_granger=False,
        run_eventstudy=False,
    )
    result = run_pipeline(cfg)
    assert result.canonical_df.empty


def test_pipeline_loads_existing_canonical_when_clustering_disabled(tmp_path):
    pool_path = _make_pool(tmp_path)
    _make_fake_canonical(tmp_path, "ML.PA")

    cfg = PipelineConfig(
        ticker="ML.PA",
        pool_path=pool_path,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        run_embedding=False,
        run_clustering=False,
        run_timeseries=False,
        run_granger=False,
        run_eventstudy=False,
    )
    result = run_pipeline(cfg)
    assert not result.canonical_df.empty
    assert "ML.PA-management-0" in result.canonical_df["canonical_id"].values


def test_pipeline_runs_eventstudy(tmp_path):
    pool_path = _make_pool(tmp_path)
    _make_fake_embeddings(tmp_path, "ML.PA")
    canonical_df = _make_fake_canonical(tmp_path, "ML.PA")

    with (
        patch("signal_intelligence.pipeline.embed_ptcs") as mock_embed,
        patch(
            "signal_intelligence.pipeline.cluster_ptcs", return_value=canonical_df
        ) as mock_cluster,
        patch("signal_intelligence.pipeline.get_prices", return_value=_fake_prices()),
        patch("signal_intelligence.pipeline.get_earnings", return_value=_fake_earnings()),
    ):
        cfg = PipelineConfig(
            ticker="ML.PA",
            pool_path=pool_path,
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
            run_sector_ontology=False,
            run_embedding=True,
            run_clustering=True,
            run_timeseries=False,
            run_granger=False,
            run_eventstudy=True,
        )
        result = run_pipeline(cfg)

    mock_embed.assert_called_once()
    mock_cluster.assert_called_once()
    assert not result.eventstudy_df.empty


def test_pipeline_runs_granger(tmp_path):
    pool_path = _make_pool(tmp_path)
    canonical_df = _make_fake_canonical(tmp_path, "ML.PA")

    # Build a non-trivial aligned_df so Granger actually runs
    quarters = pd.date_range("2019-03-31", periods=24, freq="QE")
    rng = np.random.default_rng(0)
    aligned_df = pd.DataFrame(
        [
            {
                "canonical_id": "ML.PA-management-0",
                "quarter_end": q,
                "mention_count": int(rng.integers(0, 5)),
                "surprise_pct": float(rng.standard_normal()),
            }
            for q in quarters
        ]
    )

    with (
        patch("signal_intelligence.pipeline.embed_ptcs"),
        patch("signal_intelligence.pipeline.cluster_ptcs", return_value=canonical_df),
        patch("signal_intelligence.pipeline.get_prices", return_value=_fake_prices()),
        patch("signal_intelligence.pipeline.get_earnings", return_value=_fake_earnings()),
        patch(
            "signal_intelligence.pipeline.build_timeseries",
            return_value=(pd.DataFrame(), aligned_df),
        ),
    ):
        cfg = PipelineConfig(
            ticker="ML.PA",
            pool_path=pool_path,
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
            run_sector_ontology=False,
            run_embedding=True,
            run_clustering=True,
            run_timeseries=True,
            run_granger=True,
            run_eventstudy=False,
        )
        result = run_pipeline(cfg)

    assert not result.granger_df.empty
    assert "canonical_id" in result.granger_df.columns


def test_pipeline_sector_ontology_skipped_without_company_info(tmp_path):
    pool_path = _make_pool(tmp_path)
    cfg = PipelineConfig(
        ticker="ML.PA",
        pool_path=pool_path,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        run_sector_ontology=True,  # toggled on but no company_name/sector
        run_embedding=False,
        run_clustering=False,
        run_timeseries=False,
        run_granger=False,
        run_eventstudy=False,
    )
    # Should not raise — just logs a warning and sets empty sector_vectors
    result = run_pipeline(cfg)
    assert result.sector_vectors.shape[0] == 0
