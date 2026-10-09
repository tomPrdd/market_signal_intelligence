"""Phase 6 — Temporal alignment.

Converts the canonical table into a quarterly mention-count matrix and aligns
it with the earnings surprise series from market.py.

Outputs
-------
mention_matrix : pd.DataFrame
    Index = quarter-end dates (period end, e.g. 2024-03-31).
    Columns = canonical_id strings.
    Values  = integer mention counts (number of PTCs attributed to that
               canonical in that quarter, counted by source_date).

aligned : pd.DataFrame
    Columns: [canonical_id, quarter, mention_count, surprise_pct].
    One row per (canonical, quarter) pair where both a mention count and an
    earnings surprise are available.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _quarter_end(d: pd.Timestamp) -> pd.Timestamp:
    """Return the last calendar day of the quarter containing d (midnight)."""
    return d.to_period("Q").to_timestamp("D", how="end").normalize()


def build_mention_matrix(
    canonical_df: pd.DataFrame,
    pool_path: Path,
) -> pd.DataFrame:
    """Build a quarterly mention-count matrix from the canonical table + PTC pool.

    Parameters
    ----------
    canonical_df:
        Output of cluster_ptcs(). Must have columns canonical_id and source_ids
        (JSON list of source_id strings that belong to each canonical).
    pool_path:
        Path to the JSONL PTC pool. Used to look up source_date per source_id.

    Returns
    -------
    DataFrame with quarter-end dates as index and canonical_ids as columns.
    Values are integer mention counts.
    """
    from ptc_extraction.io import load_pool

    ptcs = load_pool(pool_path)
    sid_to_date: dict[str, pd.Timestamp] = {p.source_id: pd.Timestamp(p.source_date) for p in ptcs}

    records: list[dict] = []
    for _, row in canonical_df.iterrows():
        cid = row["canonical_id"]
        try:
            source_ids: list[str] = json.loads(row["source_ids"])
        except (TypeError, ValueError):
            continue
        for sid in source_ids:
            ts = sid_to_date.get(sid)
            if ts is None:
                continue
            records.append({"canonical_id": cid, "quarter_end": _quarter_end(ts)})

    if not records:
        logger.warning("No PTC-to-canonical records found; returning empty matrix.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    matrix = df.groupby(["quarter_end", "canonical_id"]).size().unstack(fill_value=0).sort_index()
    matrix.index.name = "quarter_end"
    logger.info("Mention matrix: %d quarters x %d canonicals", len(matrix), len(matrix.columns))
    return matrix


def align_with_earnings(
    mention_matrix: pd.DataFrame,
    earnings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join mention matrix with earnings surprises on quarter.

    The earnings date is mapped to its quarter-end so it aligns with the
    mention matrix index. Only quarters present in BOTH are kept.

    Parameters
    ----------
    mention_matrix:
        Output of build_mention_matrix().
    earnings_df:
        Output of get_earnings(). Index = date (earnings announcement date),
        column surprise_pct required.

    Returns
    -------
    DataFrame with columns: canonical_id, quarter_end, mention_count,
    surprise_pct. One row per (canonical_id, quarter_end) pair.
    """
    if mention_matrix.empty or earnings_df.empty:
        return pd.DataFrame(
            columns=["canonical_id", "quarter_end", "mention_count", "surprise_pct"]
        )

    earn = earnings_df[["surprise_pct"]].copy()
    earn.index = pd.to_datetime(earn.index)
    earn["quarter_end"] = earn.index.map(_quarter_end)
    earn = earn.groupby("quarter_end")["surprise_pct"].mean()

    common_quarters = mention_matrix.index.intersection(earn.index)
    if len(common_quarters) == 0:
        logger.warning("No overlapping quarters between mention matrix and earnings.")
        return pd.DataFrame(
            columns=["canonical_id", "quarter_end", "mention_count", "surprise_pct"]
        )

    matrix_aligned = mention_matrix.loc[common_quarters]
    earn_aligned = earn.loc[common_quarters]

    rows: list[dict] = []
    for cid in matrix_aligned.columns:
        for qdate in common_quarters:
            rows.append(
                {
                    "canonical_id": cid,
                    "quarter_end": qdate,
                    "mention_count": int(matrix_aligned.at[qdate, cid]),
                    "surprise_pct": float(earn_aligned.at[qdate]),
                }
            )

    result = pd.DataFrame(rows).sort_values(["canonical_id", "quarter_end"]).reset_index(drop=True)
    logger.info(
        "Aligned dataset: %d rows (%d canonicals x %d quarters)",
        len(result),
        result["canonical_id"].nunique(),
        result["quarter_end"].nunique(),
    )
    return result


def build_timeseries(
    ticker: str,
    canonical_df: pd.DataFrame,
    pool_path: Path,
    earnings_df: pd.DataFrame,
    output_dir: Path = Path("data/processed"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Phase 6 end-to-end for a ticker.

    Returns (mention_matrix, aligned_df). Saves both to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_mention_matrix(canonical_df, pool_path)
    aligned = align_with_earnings(matrix, earnings_df)

    if not matrix.empty:
        matrix.to_csv(output_dir / f"{ticker}_mention_matrix.csv")
    if not aligned.empty:
        aligned.to_csv(output_dir / f"{ticker}_aligned.csv", index=False)

    return matrix, aligned
