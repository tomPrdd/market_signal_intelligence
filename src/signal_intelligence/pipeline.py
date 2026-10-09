"""Pipeline orchestrator — wires all phases for a single ticker.

Usage
-----
    from signal_intelligence.pipeline import run_pipeline, PipelineConfig

    cfg = PipelineConfig(
        ticker="ML.PA",
        company_name="Michelin",
        sector="Specialty Chemicals & Tires",
        pool_path=Path("data/processed/ptc_pool.jsonl"),
    )
    result = run_pipeline(cfg)
    print(result.canonical_df)
    print(result.granger_df)
    print(result.eventstudy_df)

All intermediate artefacts are written under data/interim/ and data/processed/.
Each phase is skipped if its output artefact already exists on disk, unless
refresh=True is passed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from signal_intelligence.clustering import cluster_ptcs
from signal_intelligence.correlation import run_granger_analysis
from signal_intelligence.embedding import embed_ptcs
from signal_intelligence.eventstudy import run_car_profiles, run_event_study
from signal_intelligence.market import get_earnings, get_prices
from signal_intelligence.timeseries import build_timeseries

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    # --- required ---
    ticker: str
    pool_path: Path  # JSONL pool of already-extracted PTCs

    # --- company info for sector ontology ---
    company_name: str = ""
    sector: str = ""

    # --- directories ---
    raw_dir: Path = Path("data/raw")
    market_dir: Path = Path("data/market")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")

    # --- phase toggles ---
    run_sector_ontology: bool = False  # Phase 0 (requires Exa + extra Bedrock calls)
    run_embedding: bool = True  # Phase 2
    run_clustering: bool = True  # Phases 3-5
    run_timeseries: bool = True  # Phase 6
    run_granger: bool = True  # Phase 7A
    run_eventstudy: bool = True  # Phase 7B

    # --- clustering params ---
    min_cluster_size: int = 3
    cross_map_threshold: float = 0.82
    sector_threshold: float = 0.80

    # --- granger params ---
    granger_alpha: float = 0.05
    sparsity_threshold: float = 0.30
    max_lag: int = 4

    # --- refresh flags ---
    refresh_embeddings: bool = False
    refresh_market: bool = False
    refresh_sector: bool = False

    # --- bedrock region ---
    region: str = "eu-west-3"

    # --- sector ontology (injected externally or built by Phase 0) ---
    sector_vectors: np.ndarray | None = field(default=None, repr=False)


@dataclass
class PipelineResult:
    ticker: str
    canonical_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    mention_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    aligned_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    granger_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    eventstudy_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    car_profiles_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_vectors: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 1024), dtype=np.float32)
    )


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    """Execute the full Signal Intelligence pipeline for cfg.ticker.

    Returns a PipelineResult with all DataFrames populated for phases that ran.
    """
    result = PipelineResult(ticker=cfg.ticker)

    # --- Phase 0: sector ontology ---
    sector_vectors = cfg.sector_vectors
    if cfg.run_sector_ontology and sector_vectors is None:
        if not cfg.company_name or not cfg.sector:
            logger.warning(
                "run_sector_ontology=True but company_name/sector not set — skipping Phase 0."
            )
        else:
            logger.info("[Phase 0] Building sector ontology for %s", cfg.ticker)
            from signal_intelligence.sector_ontology import build_sector_ontology

            sector_vectors = build_sector_ontology(
                focal_ticker=cfg.ticker,
                company_name=cfg.company_name,
                sector=cfg.sector,
                pool_path=cfg.pool_path,
                output_dir=cfg.interim_dir,
                region=cfg.region,
                refresh=cfg.refresh_sector,
            )
            logger.info("[Phase 0] Done — %d sector vectors.", len(sector_vectors))

    result.sector_vectors = (
        sector_vectors if sector_vectors is not None else np.zeros((0, 1024), dtype=np.float32)
    )

    # --- Phase 2: embedding ---
    if cfg.run_embedding:
        logger.info("[Phase 2] Embedding PTCs for %s", cfg.ticker)
        embed_ptcs(
            ticker=cfg.ticker,
            pool_path=cfg.pool_path,
            output_dir=cfg.interim_dir,
            region=cfg.region,
            refresh=cfg.refresh_embeddings,
        )
        logger.info("[Phase 2] Done.")

    # --- Phases 3-5: clustering + cross-map + sector tagging ---
    canonical_df = pd.DataFrame()
    if cfg.run_clustering:
        logger.info("[Phases 3-5] Clustering for %s", cfg.ticker)
        canonical_df = cluster_ptcs(
            ticker=cfg.ticker,
            embeddings_dir=cfg.interim_dir,
            output_dir=cfg.processed_dir,
            min_cluster_size=cfg.min_cluster_size,
            cross_map_threshold=cfg.cross_map_threshold,
            sector_vectors=sector_vectors,
            sector_threshold=cfg.sector_threshold,
        )
        result.canonical_df = canonical_df
        logger.info("[Phases 3-5] Done — %d canonicals.", len(canonical_df))
    else:
        # Try to load from disk if clustering was run previously
        csv_path = cfg.processed_dir / f"{cfg.ticker}_canonicals.csv"
        if csv_path.exists():
            canonical_df = pd.read_csv(csv_path)
            result.canonical_df = canonical_df

    if canonical_df.empty:
        logger.warning("No canonical table — skipping Phases 6-7.")
        return result

    # --- Market data (needed for Phases 6 and 7B) ---
    prices_df = pd.DataFrame()
    earnings_df = pd.DataFrame()
    if cfg.run_timeseries or cfg.run_eventstudy or cfg.run_granger:
        try:
            prices_df = get_prices(cfg.ticker, cache_dir=cfg.market_dir, refresh=cfg.refresh_market)
        except Exception:
            logger.exception("Failed to fetch prices for %s", cfg.ticker)

        try:
            earnings_df = get_earnings(
                cfg.ticker, cache_dir=cfg.market_dir, refresh=cfg.refresh_market
            )
        except Exception:
            logger.exception("Failed to fetch earnings for %s", cfg.ticker)

    # --- Phase 6: time series ---
    mention_matrix = pd.DataFrame()
    aligned_df = pd.DataFrame()
    if cfg.run_timeseries:
        logger.info("[Phase 6] Building mention matrix for %s", cfg.ticker)
        mention_matrix, aligned_df = build_timeseries(
            ticker=cfg.ticker,
            canonical_df=canonical_df,
            pool_path=cfg.pool_path,
            earnings_df=earnings_df,
            output_dir=cfg.processed_dir,
        )
        result.mention_matrix = mention_matrix
        result.aligned_df = aligned_df
        logger.info("[Phase 6] Done — %d rows in aligned dataset.", len(aligned_df))

    # --- Phase 7A: Granger ---
    if cfg.run_granger and not aligned_df.empty:
        logger.info("[Phase 7A] Running Granger analysis for %s", cfg.ticker)
        granger_df = run_granger_analysis(
            aligned_df=aligned_df,
            canonical_df=canonical_df,
            alpha=cfg.granger_alpha,
            sparsity_threshold=cfg.sparsity_threshold,
            max_lag=cfg.max_lag,
            output_dir=cfg.processed_dir,
            ticker=cfg.ticker,
        )
        result.granger_df = granger_df
        logger.info("[Phase 7A] Done.")

    # --- Phase 7B: event study ---
    if cfg.run_eventstudy and not prices_df.empty:
        logger.info("[Phase 7B] Running event study for %s", cfg.ticker)
        eventstudy_df = run_event_study(
            canonical_df=canonical_df,
            pool_path=cfg.pool_path,
            prices_df=prices_df,
            output_dir=cfg.processed_dir,
            ticker=cfg.ticker,
        )
        result.eventstudy_df = eventstudy_df
        result.car_profiles_df = run_car_profiles(
            canonical_df=canonical_df,
            pool_path=cfg.pool_path,
            prices_df=prices_df,
            output_dir=cfg.processed_dir,
            ticker=cfg.ticker,
        )
        logger.info("[Phase 7B] Done.")

    logger.info("Pipeline complete for %s.", cfg.ticker)
    return result
