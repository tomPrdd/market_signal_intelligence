"""Run the full Signal Intelligence pipeline for one ticker.

Fetch corpus -> extract PTCs -> run analytics pipeline -> sync artefacts to S3.

Extraction defaults to Amazon Nova Pro (BedrockBackend.DEFAULT_MODEL) — pass --model
to use Claude or another Bedrock inference profile instead.

Usage:
    uv run python scripts/run_ticker.py ML.PA
    uv run python scripts/run_ticker.py ML.PA --skip-fetch --no-s3
    uv run python scripts/run_ticker.py AAPL --model eu.anthropic.claude-sonnet-4-6
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ptc_extraction import BedrockBackend, PTCExtractor

from signal_intelligence import (
    PipelineConfig,
    acquire_corpus,
    load_documents,
    load_universe,
    run_pipeline,
    sync_ticker_to_s3,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_ticker")

POOL_PATH = Path("data/processed/ptc_pool.jsonl")

# Approximate Bedrock EU on-demand prices, USD per million tokens (input, output).
# Order matters: more specific substrings first (nova-pro before nova/nova-lite).
_PRICES_PER_MTOK = {
    "nova-pro": (0.80, 3.20),
    "nova-lite": (0.06, 0.24),
    "nova-micro": (0.035, 0.14),
    "sonnet": (3.30, 16.50),
    "haiku": (1.10, 5.50),
    "opus": (16.50, 82.50),
}


def _estimate_cost(model_id: str, usage: dict[str, int]) -> float:
    rates = next(
        (rates for name, rates in _PRICES_PER_MTOK.items() if name in model_id),
        _PRICES_PER_MTOK["sonnet"],
    )
    return usage["input_tokens"] / 1e6 * rates[0] + usage["output_tokens"] / 1e6 * rates[1]


def run_ticker(
    ticker: str,
    universe_path: Path = Path("configs/universe.yaml"),
    model_id: str | None = None,
    skip_fetch: bool = False,
    skip_extract: bool = False,
    sync_s3: bool = True,
    max_news: int = 250,
    max_workers: int = 8,
) -> dict:
    """Full run for one ticker. Returns a stats dict (counts, tokens, cost)."""
    companies = {c["ticker"]: c for c in load_universe(universe_path)}
    if ticker not in companies:
        raise SystemExit(f"Ticker {ticker!r} not in {universe_path}")
    company = companies[ticker]

    stats: dict = {"ticker": ticker}

    if not skip_fetch:
        logger.info("=== Acquiring corpus for %s ===", ticker)
        stats["acquisition"] = acquire_corpus(company, max_news=max_news)

    if not skip_extract:
        logger.info("=== Extracting PTCs for %s ===", ticker)
        documents = load_documents(ticker)
        if not documents:
            logger.warning("No documents found for %s — nothing to extract.", ticker)
        else:
            backend = BedrockBackend(model_id=model_id)
            extractor = PTCExtractor(backend, max_workers=max_workers)
            stats["extraction"] = extractor.extract_from_documents(
                documents, output_path=POOL_PATH, resume=True
            )
            usage = backend.usage
            cost = _estimate_cost(backend.model_id, usage)
            stats["tokens"] = usage
            stats["cost_usd"] = round(cost, 2)
            logger.info(
                "=== LLM usage: %d in / %d out tokens, ~$%.2f (%s) ===",
                usage["input_tokens"],
                usage["output_tokens"],
                cost,
                backend.model_id,
            )

    logger.info("=== Running analytics pipeline for %s ===", ticker)
    cfg = PipelineConfig(
        ticker=ticker,
        company_name=company.get("name", ticker),
        sector=company.get("sector", ""),
        pool_path=POOL_PATH,
    )
    result = run_pipeline(cfg)
    stats["canonicals"] = len(result.canonical_df)
    stats["eventstudy_rows"] = len(result.eventstudy_df)
    stats["car_profile_rows"] = len(result.car_profiles_df)

    if sync_s3:
        logger.info("=== Syncing %s artefacts to S3 ===", ticker)
        stats["s3"] = sync_ticker_to_s3(ticker)

    logger.info("=== Done: %s ===", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", help="Ticker from configs/universe.yaml (e.g. ML.PA)")
    parser.add_argument("--universe", type=Path, default=Path("configs/universe.yaml"))
    parser.add_argument("--model", default=None, help="Bedrock model id for extraction")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip corpus acquisition")
    parser.add_argument("--skip-extract", action="store_true", help="Skip PTC extraction")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 sync")
    parser.add_argument("--max-news", type=int, default=250)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent extraction threads (lower on small/constrained instances)",
    )
    args = parser.parse_args()

    run_ticker(
        args.ticker,
        universe_path=args.universe,
        model_id=args.model,
        skip_fetch=args.skip_fetch,
        skip_extract=args.skip_extract,
        sync_s3=not args.no_s3,
        max_news=args.max_news,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
