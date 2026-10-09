"""Data loaders for the Streamlit app.

Resolution order per artefact:
1. data/processed/ and data/market/ — fresh local pipeline runs (development)
2. app/demo/ — the bundled demo dataset shipped with the deployed app
3. SI_DATA_URL — optional HTTPS base (e.g. a public S3 prefix) for remote data

The deployed demo app relies only on (2): everything it shows is baked into
the image by scripts/build_demo_bundle.py, so it needs no credentials and no
network at runtime.
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

PROCESSED = Path("data/processed")
MARKET = Path("data/market")
DEMO = Path(__file__).resolve().parent / "demo"


def _remote_base() -> str | None:
    if os.environ.get("SI_DATA_URL"):
        return os.environ["SI_DATA_URL"]
    try:
        return st.secrets.get("SI_DATA_URL", None)
    except Exception:  # no secrets.toml configured
        return None


def _read_csv(candidates: list[Path], remote_key: str, **kwargs) -> pd.DataFrame:
    for local in candidates:
        if local.exists():
            return pd.read_csv(local, **kwargs)
    base = _remote_base()
    if base:
        try:
            return pd.read_csv(f"{base.rstrip('/')}/{remote_key}", **kwargs)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def available_tickers() -> list[str]:
    """Tickers with a canonical table available anywhere."""
    tickers = {p.name.removesuffix("_canonicals.csv") for p in PROCESSED.glob("*_canonicals.csv")}
    tickers |= {p.name.removesuffix("_canonicals.csv") for p in DEMO.glob("*_canonicals.csv")}
    base = _remote_base()
    if base:
        try:
            manifest = pd.read_csv(f"{base.rstrip('/')}/processed/manifest.csv")
            tickers.update(manifest["ticker"].astype(str))
        except Exception:
            pass
    return sorted(tickers)


def _processed_candidates(name: str) -> list[Path]:
    return [PROCESSED / name, DEMO / name]


@st.cache_data(ttl=3600)
def load_canonicals(ticker: str) -> pd.DataFrame:
    df = _read_csv(
        _processed_candidates(f"{ticker}_canonicals.csv"),
        f"processed/{ticker}/{ticker}_canonicals.csv",
    )
    if not df.empty and "is_noise" in df.columns:
        df = df[~df["is_noise"].astype(str).str.lower().eq("true")]
    return df


@st.cache_data(ttl=3600)
def load_eventstudy(ticker: str) -> pd.DataFrame:
    return _read_csv(
        _processed_candidates(f"{ticker}_eventstudy.csv"),
        f"processed/{ticker}/{ticker}_eventstudy.csv",
    )


@st.cache_data(ttl=3600)
def load_car_profiles(ticker: str) -> pd.DataFrame:
    return _read_csv(
        _processed_candidates(f"{ticker}_car_profiles.csv"),
        f"processed/{ticker}/{ticker}_car_profiles.csv",
    )


@st.cache_data(ttl=3600)
def load_granger(ticker: str) -> pd.DataFrame:
    return _read_csv(
        _processed_candidates(f"{ticker}_granger.csv"),
        f"processed/{ticker}/{ticker}_granger.csv",
    )


@st.cache_data(ttl=3600)
def load_mention_matrix(ticker: str) -> pd.DataFrame:
    df = _read_csv(
        _processed_candidates(f"{ticker}_mention_matrix.csv"),
        f"processed/{ticker}/{ticker}_mention_matrix.csv",
        index_col=0,
    )
    if not df.empty:
        df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=3600)
def load_prices(ticker: str) -> pd.DataFrame:
    df = _read_csv(
        [MARKET / f"{ticker}_prices.csv", DEMO / f"{ticker}_prices.csv"],
        f"market/{ticker}_prices.csv",
        index_col="date",
    )
    if not df.empty:
        df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=3600)
def load_robustness(ticker: str) -> pd.DataFrame:
    """Per-mechanism robustness columns (market model, BH p-values, distinct dates).

    Produced offline by scripts/export_robustness.py. Empty when not available.
    """
    return _read_csv(
        [PROCESSED / f"{ticker}_robustness.csv", DEMO / f"{ticker}_robustness.csv"],
        f"processed/{ticker}/{ticker}_robustness.csv",
    )


@st.cache_data(ttl=3600)
def load_projection(ticker: str) -> pd.DataFrame:
    """3D UMAP coordinates of every claim (built offline by build_projection.py)."""
    return _read_csv(
        [PROCESSED / f"{ticker}_projection.csv", DEMO / f"{ticker}_projection.csv"],
        f"processed/{ticker}/{ticker}_projection.csv",
    )


@st.cache_data(ttl=3600)
def load_cluster_summaries(ticker: str) -> pd.DataFrame:
    """LLM-generalized mechanism per top cluster (built by build_cluster_summaries.py)."""
    return _read_csv(
        [PROCESSED / f"{ticker}_cluster_summaries.csv", DEMO / f"{ticker}_cluster_summaries.csv"],
        f"processed/{ticker}/{ticker}_cluster_summaries.csv",
    )


@st.cache_data(ttl=3600)
def load_sources(ticker: str) -> pd.DataFrame:
    """Document inventory produced by scripts/build_demo_bundle.py."""
    df = _read_csv(
        [DEMO / f"{ticker}_sources.csv"],
        f"processed/{ticker}/{ticker}_sources.csv",
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_ptcs(ticker: str) -> pd.DataFrame:
    """PTC records for one ticker, from the shared pool or the bundled subset."""
    lines: list[str] = []
    for candidate in (PROCESSED / "ptc_pool.jsonl", DEMO / f"{ticker}_ptc_pool.jsonl"):
        if candidate.exists():
            lines = candidate.read_text(encoding="utf-8").splitlines()
            break
    if not lines:
        base = _remote_base()
        if base:
            try:
                import requests

                resp = requests.get(f"{base.rstrip('/')}/processed/ptc_pool.jsonl", timeout=30)
                resp.raise_for_status()
                lines = resp.text.splitlines()
            except Exception:
                return pd.DataFrame()

    records = []
    prefix = f"{ticker}-"
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(rec.get("source_id", "")).startswith(prefix):
            records.append(rec)
    df = pd.DataFrame(records)
    if not df.empty:
        df["source_date"] = pd.to_datetime(df["source_date"], errors="coerce")
    return df


def canonical_label(row: pd.Series, max_len: int = 90) -> str:
    """Short human-readable label for a canonical mechanism."""
    mech = str(row.get("representative_mechanism", row.get("canonical_id", "")))
    return mech if len(mech) <= max_len else mech[: max_len - 1] + "…"
