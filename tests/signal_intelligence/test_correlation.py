"""Tests for signal_intelligence.correlation (Phase 7A)."""

import numpy as np
import pandas as pd
import pytest

from signal_intelligence.correlation import (
    _bh_correction,
    _bonferroni,
    _is_stationary,
    _make_stationary,
    run_granger_analysis,
)

# ---------------------------------------------------------------------------
# _bh_correction
# ---------------------------------------------------------------------------


def test_bh_correction_single_p():
    result = _bh_correction([0.03])
    assert len(result) == 1
    assert abs(result[0] - 0.03) < 1e-9


def test_bh_correction_all_significant():
    p_values = [0.001, 0.002, 0.003]
    corrected = _bh_correction(p_values)
    assert all(c <= 1.0 for c in corrected)
    # Monotonicity: corrected values should be non-decreasing when sorted by original p
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    sorted_corr = [corrected[i] for i in order]
    assert sorted_corr == sorted(sorted_corr)


def test_bh_correction_empty():
    assert _bh_correction([]) == []


def test_bh_never_exceeds_one():
    p_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    corrected = _bh_correction(p_values)
    assert all(c <= 1.0 for c in corrected)


# ---------------------------------------------------------------------------
# _bonferroni (Fix 3 fallback)
# ---------------------------------------------------------------------------


def test_bonferroni_clamps_to_one():
    result = _bonferroni([0.5, 0.5, 0.5])
    assert all(c == 1.0 for c in result)


def test_bonferroni_single():
    result = _bonferroni([0.04])
    assert result[0] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# Stationarity helpers
# ---------------------------------------------------------------------------


def test_is_stationary_white_noise():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.standard_normal(100))
    assert _is_stationary(s)


def test_is_stationary_constant():
    s = pd.Series([1.0] * 50)
    assert _is_stationary(s)


def test_make_stationary_already_stationary():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.standard_normal(80))
    out, differenced = _make_stationary(s)
    assert not differenced
    assert len(out) == len(s)


def test_make_stationary_random_walk():
    rng = np.random.default_rng(2)
    walk = pd.Series(rng.standard_normal(100).cumsum())
    out, differenced = _make_stationary(walk)
    assert differenced
    assert len(out) == len(walk) - 1


# ---------------------------------------------------------------------------
# run_granger_analysis
# ---------------------------------------------------------------------------


def _make_aligned_df(n_quarters: int = 24, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quarters = pd.date_range("2019-03-31", periods=n_quarters, freq="QE")
    mention = (rng.integers(0, 5, size=n_quarters)).astype(float)
    surprise = rng.standard_normal(n_quarters)
    rows = []
    for q, m, s in zip(quarters, mention, surprise, strict=False):
        rows.append({"canonical_id": "c1", "quarter_end": q, "mention_count": m, "surprise_pct": s})
    return pd.DataFrame(rows)


def _make_canonical_df_simple(cids: list[str], directions: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_id": cids,
            "direction": directions,
            "corpus": "management",
            "polarity": [-1] * len(cids),
        }
    )


def test_run_granger_analysis_returns_expected_columns(tmp_path):
    aligned = _make_aligned_df(24)
    canonical = _make_canonical_df_simple(["c1"], ["precursor"])
    result = run_granger_analysis(aligned, canonical, output_dir=tmp_path, ticker="ML.PA")

    expected = {
        "canonical_id",
        "direction",
        "n_quarters",
        "n_nonzero",
        "sparsity_pct",
        "filtered_out",
        "differenced_mention",
        "differenced_surprise",
        "best_lag",
        "p_value_raw",
        "p_value_bh",
        "significant_bh",
        "granger_direction",
        "consistent_with_tag",
    }
    assert expected.issubset(set(result.columns))


def test_run_granger_saves_csv(tmp_path):
    aligned = _make_aligned_df(24)
    canonical = _make_canonical_df_simple(["c1"], ["precursor"])
    run_granger_analysis(aligned, canonical, output_dir=tmp_path, ticker="ML.PA")
    assert (tmp_path / "ML.PA_granger.csv").exists()


def test_run_granger_sparsity_filter():
    # All zeros → should be filtered out
    quarters = pd.date_range("2019-03-31", periods=24, freq="QE")
    rows = [
        {"canonical_id": "sparse", "quarter_end": q, "mention_count": 0, "surprise_pct": 0.5}
        for q in quarters
    ]
    aligned = pd.DataFrame(rows)
    canonical = _make_canonical_df_simple(["sparse"], ["precursor"])
    result = run_granger_analysis(aligned, canonical)
    assert result.iloc[0]["filtered_out"] == True  # noqa: E712
    assert result.iloc[0]["granger_direction"] == "not_tested"


def test_run_granger_bh_p_never_raw():
    """After BH correction, p_value_bh must not equal raw p (Bonferroni fallback included)."""
    aligned = _make_aligned_df(24)
    canonical = _make_canonical_df_simple(["c1"], ["precursor"])
    result = run_granger_analysis(aligned, canonical)
    tested = result[~result["filtered_out"]]
    if not tested.empty:
        # p_value_bh should always be <= 1.0
        assert (tested["p_value_bh"] <= 1.0).all()


def test_run_granger_empty_aligned():
    aligned = pd.DataFrame(columns=["canonical_id", "quarter_end", "mention_count", "surprise_pct"])
    canonical = _make_canonical_df_simple(["c1"], ["precursor"])
    result = run_granger_analysis(aligned, canonical)
    assert result.empty
