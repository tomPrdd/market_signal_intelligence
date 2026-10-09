"""Run the pipeline over the whole universe (or a subset), one ticker at a time.

Each ticker is isolated: a failure logs and moves on. Prints a cost summary at
the end. Everything is resume-safe, so re-running only does missing work.

Usage:
    uv run python scripts/run_universe.py --tickers ML.PA AAPL MSFT
    uv run python scripts/run_universe.py --limit 10 --model eu.anthropic.claude-haiku-4-5-20251001-v1:0
    uv run python scripts/run_universe.py            # full universe
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from run_ticker import run_ticker

from signal_intelligence import load_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_universe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("configs/universe.yaml"))
    parser.add_argument("--tickers", nargs="*", default=None, help="Subset of tickers to run")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N companies")
    parser.add_argument("--model", default=None, help="Bedrock model id for extraction")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--no-s3", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent extraction threads (lower on small/constrained instances)",
    )
    args = parser.parse_args()

    companies = load_universe(args.universe)
    tickers = [c["ticker"] for c in companies]
    if args.tickers:
        tickers = [t for t in tickers if t in set(args.tickers)]
    if args.limit:
        tickers = tickers[: args.limit]

    results: list[dict] = []
    failures: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        logger.info("########## [%d/%d] %s ##########", i, len(tickers), ticker)
        try:
            stats = run_ticker(
                ticker,
                universe_path=args.universe,
                model_id=args.model,
                skip_fetch=args.skip_fetch,
                sync_s3=not args.no_s3,
                max_workers=args.max_workers,
            )
            results.append(stats)
        except Exception:
            logger.exception("Ticker %s failed — continuing.", ticker)
            failures.append(ticker)

    total_cost = sum(s.get("cost_usd", 0.0) for s in results)
    logger.info("=" * 60)
    logger.info("Universe run complete: %d ok, %d failed.", len(results), len(failures))
    if failures:
        logger.info("Failed tickers: %s", ", ".join(failures))
    logger.info("Total estimated LLM cost: $%.2f", total_cost)
    for s in results:
        logger.info(
            "  %-10s canonicals=%-4s cost=$%-8s",
            s["ticker"],
            s.get("canonicals", "-"),
            s.get("cost_usd", 0.0),
        )


if __name__ == "__main__":
    main()
