"""Add a press corpus to companies that only have management docs.

For each ticker: fetch GDELT press if none is collected yet, then extract PTCs
from the PRESS documents only (management is already in the pool from earlier
runs), and re-run the analytics pipeline so mechanisms get the company-vs-press
triangulation. Loading only press (source_types=['press']) skips re-parsing the
large management PDFs entirely — that parsing is what timed out before.

Usage:
    uv run python scripts/add_press.py TTE.PA OR.PA SAN.PA MC.PA
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ptc_extraction import BedrockBackend, PTCExtractor

from signal_intelligence import (
    PipelineConfig,
    load_universe,
    run_pipeline,
)
from signal_intelligence.acquisition import fetch_gdelt_press
from signal_intelligence.ingestion import load_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("add_press")

POOL = Path("data/processed/ptc_pool.jsonl")


def add_press(ticker: str, max_workers: int = 6) -> dict:
    companies = {c["ticker"]: c for c in load_universe()}
    company = companies[ticker]
    name = company.get("name", ticker)

    press_dir = Path(f"data/raw/{ticker}/press")
    n_press_files = len(list(press_dir.glob("*.txt"))) if press_dir.exists() else 0
    if n_press_files == 0:
        logger.info("=== [%s] No press collected — fetching from GDELT ===", ticker)
        fetch_gdelt_press(ticker, name)
        n_press_files = len(list(press_dir.glob("*.txt"))) if press_dir.exists() else 0
    logger.info("=== [%s] %d press files on disk ===", ticker, n_press_files)

    press_docs = load_documents(ticker, source_types=["press"])
    stats: dict = {"ticker": ticker, "press_files": n_press_files, "cost_usd": 0.0}
    if press_docs:
        logger.info("=== [%s] Extracting PTCs from %d press docs ===", ticker, len(press_docs))
        backend = BedrockBackend()
        extractor = PTCExtractor(backend, max_workers=max_workers)
        stats["extraction"] = extractor.extract_from_documents(
            press_docs, output_path=POOL, resume=True
        )
        u = backend.usage
        stats["cost_usd"] = round(
            u["input_tokens"] / 1e6 * 0.80 + u["output_tokens"] / 1e6 * 3.20, 2
        )
        logger.info("=== [%s] Press extraction cost ~$%.2f ===", ticker, stats["cost_usd"])
    else:
        logger.warning("=== [%s] No press docs to extract ===", ticker)

    logger.info("=== [%s] Re-running analytics pipeline ===", ticker)
    result = run_pipeline(
        PipelineConfig(
            ticker=ticker,
            company_name=name,
            sector=company.get("sector", ""),
            pool_path=POOL,
        )
    )
    stats["canonicals"] = len(result.canonical_df)
    if not result.canonical_df.empty and "corpus_presence" in result.canonical_df.columns:
        stats["presence"] = result.canonical_df["corpus_presence"].value_counts().to_dict()
    logger.info("=== [%s] DONE: %s ===", ticker, stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()
    for t in args.tickers:
        try:
            add_press(t, max_workers=args.max_workers)
        except Exception:
            logger.exception("[%s] failed — continuing", t)
        time.sleep(2)
    logger.info("ADD-PRESS BATCH DONE")


if __name__ == "__main__":
    main()
