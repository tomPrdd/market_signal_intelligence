import logging
import os
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path("data/market")

_FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _fmp_api_key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise OSError("FMP_API_KEY is not set. Add it to your .env file: FMP_API_KEY=your_key_here")
    return key


def get_prices(
    ticker: str,
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV DataFrame for ticker.

    Columns: date (index), open, high, low, close, volume, adj_close.
    Fetches from yfinance on first call; reads from CSV cache on subsequent calls.
    Pass refresh=True to force a re-fetch.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}_prices.csv"

    if cache_path.exists() and not refresh:
        logger.info("Loading prices from cache: %s", cache_path)
        df = pd.read_csv(cache_path, index_col="date", parse_dates=True)
        df.index = pd.to_datetime(df.index).date
        return df

    logger.info("Fetching prices from yfinance: %s", ticker)
    raw = yf.download(ticker, period="max", progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for ticker {ticker!r}")

    # yfinance returns MultiIndex columns when auto_adjust=False; flatten them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower().replace(" ", "_") for col in raw.columns]
    else:
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    raw.index.name = "date"
    rename = {
        "adj_close": "adj_close",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    raw = raw.rename(columns=rename)
    keep = [c for c in ["open", "high", "low", "close", "volume", "adj_close"] if c in raw.columns]
    df = raw[keep].copy()
    df.index = pd.to_datetime(df.index).date

    df.to_csv(cache_path, index_label="date")
    logger.info("Cached prices to %s (%d rows)", cache_path, len(df))
    return df


def _earnings_from_fmp(ticker: str) -> pd.DataFrame:
    key = _fmp_api_key()
    url = f"{_FMP_BASE}/earnings-surprises/{ticker}"
    logger.info("Fetching earnings from FMP: %s", url)
    resp = requests.get(url, params={"apikey": key}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list) or not data:
        raise ValueError(f"FMP returned no earnings data for ticker {ticker!r}: {data}")

    rows = []
    for item in data:
        rows.append(
            {
                "date": item.get("date"),
                "reported_eps": item.get("actualEarningResult"),
                "estimated_eps": item.get("estimatedEarning"),
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df["surprise_pct"] = (
        (df["reported_eps"] - df["estimated_eps"]) / df["estimated_eps"].abs() * 100
    )
    return df


def _earnings_from_yfinance(ticker: str, limit: int = 40) -> pd.DataFrame:
    """Earnings surprises from Yahoo Finance — free fallback, works for EU tickers."""
    logger.info("Fetching earnings from yfinance: %s", ticker)
    raw = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no earnings dates for ticker {ticker!r}")

    df = pd.DataFrame(
        {
            "reported_eps": raw.get("Reported EPS"),
            "estimated_eps": raw.get("EPS Estimate"),
        }
    )
    df.index = pd.to_datetime(raw.index).date
    df.index.name = "date"
    df = df.dropna(subset=["reported_eps"]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df["surprise_pct"] = (
        (df["reported_eps"] - df["estimated_eps"]) / df["estimated_eps"].abs() * 100
    )
    return df


def get_earnings(
    ticker: str,
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return earnings surprise DataFrame for ticker.

    Columns: date (index), reported_eps, estimated_eps, surprise_pct.
    Tries FMP first (requires FMP_API_KEY and a plan covering the endpoint),
    then falls back to yfinance earnings dates. CSV-cached; pass refresh=True
    to force a re-fetch.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}_earnings.csv"

    if cache_path.exists() and not refresh:
        logger.info("Loading earnings from cache: %s", cache_path)
        df = pd.read_csv(cache_path, index_col="date", parse_dates=True)
        df.index = pd.to_datetime(df.index).date
        return df

    try:
        df = _earnings_from_fmp(ticker)
    except Exception as exc:
        logger.warning(
            "FMP earnings unavailable for %s (%s); falling back to yfinance", ticker, exc
        )
        df = _earnings_from_yfinance(ticker)

    if df.empty:
        raise ValueError(f"No earnings data available for ticker {ticker!r}")

    df.to_csv(cache_path, index_label="date")
    logger.info("Cached earnings to %s (%d rows)", cache_path, len(df))
    return df
