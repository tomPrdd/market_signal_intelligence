from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from signal_intelligence.market import get_earnings, get_prices


def _make_price_df() -> pd.DataFrame:
    idx = pd.Index([date(2024, 1, 2), date(2024, 1, 3)], name="date")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1_000_000, 1_100_000],
            "adj_close": [101.0, 102.0],
        },
        index=idx,
    )


def _make_earnings_json() -> list[dict]:
    return [
        {"date": "2024-01-30", "actualEarningResult": 1.5, "estimatedEarning": 1.3},
        {"date": "2023-10-30", "actualEarningResult": 1.2, "estimatedEarning": 1.25},
    ]


# ---------------------------------------------------------------------------
# get_prices
# ---------------------------------------------------------------------------


def test_get_prices_fetches_and_caches(tmp_path):
    df_raw = _make_price_df()
    with patch("signal_intelligence.market.yf.download", return_value=df_raw) as mock_dl:
        df = get_prices("ML.PA", cache_dir=tmp_path)

    mock_dl.assert_called_once()
    assert len(df) == 2
    assert (tmp_path / "ML.PA_prices.csv").exists()


def test_get_prices_uses_cache_on_second_call(tmp_path):
    df_raw = _make_price_df()
    with patch("signal_intelligence.market.yf.download", return_value=df_raw) as mock_dl:
        get_prices("ML.PA", cache_dir=tmp_path)
        get_prices("ML.PA", cache_dir=tmp_path)

    # yfinance should only be called once — second call hits cache
    assert mock_dl.call_count == 1


def test_get_prices_refresh_bypasses_cache(tmp_path):
    df_raw = _make_price_df()
    with patch("signal_intelligence.market.yf.download", return_value=df_raw) as mock_dl:
        get_prices("ML.PA", cache_dir=tmp_path)
        get_prices("ML.PA", cache_dir=tmp_path, refresh=True)

    assert mock_dl.call_count == 2


def test_get_prices_empty_response_raises(tmp_path):
    with (
        patch("signal_intelligence.market.yf.download", return_value=pd.DataFrame()),
        pytest.raises(ValueError, match="no data"),
    ):
        get_prices("INVALID", cache_dir=tmp_path)


def test_get_prices_returns_expected_columns(tmp_path):
    df_raw = _make_price_df()
    with patch("signal_intelligence.market.yf.download", return_value=df_raw):
        df = get_prices("ML.PA", cache_dir=tmp_path)

    assert "close" in df.columns
    assert "adj_close" in df.columns


# ---------------------------------------------------------------------------
# get_earnings
# ---------------------------------------------------------------------------


def _mock_fmp_response(data: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


def test_get_earnings_fetches_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    mock_resp = _mock_fmp_response(_make_earnings_json())
    with patch("signal_intelligence.market.requests.get", return_value=mock_resp) as mock_get:
        df = get_earnings("ML.PA", cache_dir=tmp_path)

    mock_get.assert_called_once()
    assert len(df) == 2
    assert (tmp_path / "ML.PA_earnings.csv").exists()


def test_get_earnings_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    mock_resp = _mock_fmp_response(_make_earnings_json())
    with patch("signal_intelligence.market.requests.get", return_value=mock_resp) as mock_get:
        get_earnings("ML.PA", cache_dir=tmp_path)
        get_earnings("ML.PA", cache_dir=tmp_path)

    assert mock_get.call_count == 1


def test_get_earnings_refresh_bypasses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    mock_resp = _mock_fmp_response(_make_earnings_json())
    with patch("signal_intelligence.market.requests.get", return_value=mock_resp) as mock_get:
        get_earnings("ML.PA", cache_dir=tmp_path)
        get_earnings("ML.PA", cache_dir=tmp_path, refresh=True)

    assert mock_get.call_count == 2


def _mock_yf_earnings(dates: list[date], reported: list[float], estimated: list[float]):
    """Build a mock yf.Ticker whose get_earnings_dates returns a Yahoo-shaped frame."""
    raw = pd.DataFrame(
        {"Reported EPS": reported, "EPS Estimate": estimated},
        index=pd.DatetimeIndex(pd.to_datetime(dates)),
    )
    ticker = MagicMock()
    ticker.get_earnings_dates.return_value = raw
    return MagicMock(return_value=ticker)


def test_get_earnings_missing_key_falls_back_to_yfinance(tmp_path, monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    mock_ticker = _mock_yf_earnings([date(2024, 1, 30)], [1.5], [1.3])
    with patch("signal_intelligence.market.yf.Ticker", mock_ticker):
        df = get_earnings("ML.PA", cache_dir=tmp_path)

    assert len(df) == 1
    assert abs(df.iloc[0]["surprise_pct"] - (0.2 / 1.3 * 100)) < 0.01
    assert (tmp_path / "ML.PA_earnings.csv").exists()


def test_get_earnings_surprise_pct_computed(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    mock_resp = _mock_fmp_response(_make_earnings_json())
    with patch("signal_intelligence.market.requests.get", return_value=mock_resp):
        df = get_earnings("ML.PA", cache_dir=tmp_path)

    assert "surprise_pct" in df.columns
    # 2024-01-30: (1.5 - 1.3) / 1.3 * 100 ≈ 15.38
    row = df.loc[date(2024, 1, 30)]
    assert abs(row["surprise_pct"] - (0.2 / 1.3 * 100)) < 0.01


def test_get_earnings_empty_everywhere_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    mock_resp = _mock_fmp_response([])
    empty_ticker = MagicMock()
    empty_ticker.get_earnings_dates.return_value = pd.DataFrame()
    with (
        patch("signal_intelligence.market.requests.get", return_value=mock_resp),
        patch("signal_intelligence.market.yf.Ticker", MagicMock(return_value=empty_ticker)),
        pytest.raises(ValueError, match="no earnings dates"),
    ):
        get_earnings("ML.PA", cache_dir=tmp_path)
