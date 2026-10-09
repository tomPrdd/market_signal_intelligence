"""Phase 7A — Granger causality analysis on earnings surprises.

All three mandatory methodological fixes are applied here from the start:
  Fix 1: ADF stationarity test + auto-differencing before Granger.
  Fix 2: Sparsity filter — drop canonicals with <30% non-zero quarters.
  Fix 3: BH FDR correction; fallback is Bonferroni (min(p*n, 1.0)), never raw p.

Output schema (one row per canonical tested)
--------------------------------------------
canonical_id          str
direction             str  (precursor | consequence)
n_quarters            int  (total quarters in the aligned series)
n_nonzero             int  (quarters with at least one mention)
sparsity_pct          float (n_nonzero / n_quarters * 100)
filtered_out          bool (True if sparsity filter removed this canonical)
differenced_mention   bool (True if mention series was differenced for stationarity)
differenced_surprise  bool (True if surprise series was differenced)
best_lag              int  (lag with lowest p-value among tested lags)
p_value_raw           float
p_value_bh            float (BH-corrected; Bonferroni fallback if BH errors)
significant_bh        bool (p_value_bh < alpha)
granger_direction     str  ("mention_leads" | "surprise_leads" | "not_tested")
consistent_with_tag   bool | None  (does direction tag match granger_direction?)
"""

import logging
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

logger = logging.getLogger(__name__)

_SPARSITY_THRESHOLD = 0.30  # minimum fraction of non-zero quarters
_ADF_SIGNIFICANCE = 0.05
_GRANGER_ALPHA = 0.05
_MAX_LAG = 4  # tested lags: 1..MAX_LAG


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_stationary(series: pd.Series) -> bool:
    if series.std() == 0:
        return True  # constant series — trivially stationary (will be filtered anyway)
    try:
        result = adfuller(series.dropna(), autolag="AIC")
        return result[1] < _ADF_SIGNIFICANCE
    except Exception:
        return True  # conservative: don't difference if ADF errors


def _make_stationary(series: pd.Series) -> tuple[pd.Series, bool]:
    """Return (series, was_differenced)."""
    if _is_stationary(series):
        return series, False
    diffed = series.diff().dropna()
    return diffed, True


def _bh_correction(p_values: list[float], alpha: float = _GRANGER_ALPHA) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns corrected p-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [1.0] * n
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        corrected[orig_idx] = min(p * n / rank, 1.0)
    # Enforce monotonicity (standard BH step-up procedure)
    order = [i for i, _ in sorted(enumerate(p_values), key=lambda x: x[1])]
    for k in range(len(order) - 2, -1, -1):
        corrected[order[k]] = min(corrected[order[k]], corrected[order[k + 1]])
    return corrected


def _bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def _granger_best_lag(mention: pd.Series, surprise: pd.Series, max_lag: int) -> tuple[int, float]:
    """Run Granger test for lags 1..max_lag; return (best_lag, best_p_value).

    'mention Granger-causes surprise' hypothesis (mention leads).
    """
    try:
        data = pd.concat([surprise, mention], axis=1).dropna()
        if len(data) <= max_lag + 2:
            return 1, 1.0
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        best_lag, best_p = 1, 1.0
        for lag, res in results.items():
            p = res[0]["ssr_ftest"][1]  # F-test p-value
            if p < best_p:
                best_p = p
                best_lag = lag
        return best_lag, best_p
    except Exception:
        logger.exception("Granger test failed")
        return 1, 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_granger_analysis(
    aligned_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    alpha: float = _GRANGER_ALPHA,
    sparsity_threshold: float = _SPARSITY_THRESHOLD,
    max_lag: int = _MAX_LAG,
    output_dir: Path | None = None,
    ticker: str = "",
) -> pd.DataFrame:
    """Run the full Track A pipeline on the aligned mention/surprise dataset.

    Parameters
    ----------
    aligned_df:
        Output of align_with_earnings(). Columns: canonical_id, quarter_end,
        mention_count, surprise_pct.
    canonical_df:
        Output of cluster_ptcs(). Used to look up direction tag per canonical.
    alpha:
        Significance threshold applied after BH correction.
    sparsity_threshold:
        Minimum fraction of non-zero quarters; canonicals below this are
        flagged filtered_out=True and not tested.
    max_lag:
        Maximum Granger lag to test (1..max_lag).
    output_dir:
        If provided, saves results CSV to output_dir/{ticker}_granger.csv.
    ticker:
        Used for the output filename.

    Returns
    -------
    DataFrame with one row per canonical, schema described in module docstring.
    """
    direction_map: dict[str, str] = dict(
        zip(canonical_df["canonical_id"], canonical_df["direction"], strict=False)
    )

    canonical_ids = aligned_df["canonical_id"].unique().tolist()
    rows: list[dict] = []

    for cid in canonical_ids:
        sub = aligned_df[aligned_df["canonical_id"] == cid].set_index("quarter_end").sort_index()
        mention = sub["mention_count"].astype(float)
        surprise = sub["surprise_pct"].astype(float)

        n_quarters = len(mention)
        n_nonzero = int((mention > 0).sum())
        sparsity_pct = n_nonzero / n_quarters * 100 if n_quarters > 0 else 0.0
        filtered_out = sparsity_pct < sparsity_threshold * 100
        direction = direction_map.get(cid, "unknown")

        rows.append(
            {
                "canonical_id": cid,
                "direction": direction,
                "n_quarters": n_quarters,
                "n_nonzero": n_nonzero,
                "sparsity_pct": round(sparsity_pct, 1),
                "filtered_out": filtered_out,
                "differenced_mention": False,
                "differenced_surprise": False,
                "best_lag": None,
                "p_value_raw": None,
                "p_value_bh": None,
                "significant_bh": False,
                "granger_direction": "not_tested",
                "consistent_with_tag": None,
            }
        )

        if filtered_out:
            continue

        mention_s, diff_m = _make_stationary(mention)
        surprise_s, diff_sur = _make_stationary(surprise)
        rows[-1]["differenced_mention"] = diff_m
        rows[-1]["differenced_surprise"] = diff_sur

        # Align after possible differencing
        common_idx = mention_s.index.intersection(surprise_s.index)
        mention_s = mention_s.loc[common_idx]
        surprise_s = surprise_s.loc[common_idx]

        lag, p_raw = _granger_best_lag(mention_s, surprise_s, max_lag)
        rows[-1]["best_lag"] = lag
        rows[-1]["p_value_raw"] = round(p_raw, 6)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    # BH correction across all tested canonicals (Fix 3)
    tested_mask = ~result_df["filtered_out"]
    tested_p = result_df.loc[tested_mask, "p_value_raw"].tolist()
    if tested_p:
        try:
            corrected = _bh_correction(tested_p, alpha)
        except Exception:
            logger.warning("BH correction failed, falling back to Bonferroni")
            corrected = _bonferroni(tested_p)
        result_df.loc[tested_mask, "p_value_bh"] = [round(p, 6) for p in corrected]
    else:
        corrected = []

    result_df.loc[tested_mask, "significant_bh"] = result_df.loc[tested_mask, "p_value_bh"] < alpha

    # Granger direction + consistency check
    for idx in result_df.index[tested_mask]:
        direction = result_df.at[idx, "direction"]
        result_df.at[idx, "granger_direction"] = "mention_leads"
        if result_df.at[idx, "significant_bh"]:
            consistent = direction == "precursor"
            result_df.at[idx, "consistent_with_tag"] = consistent

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{ticker}_granger.csv"
        result_df.to_csv(out_path, index=False)
        logger.info("Saved Granger results to %s", out_path)

    n_sig = int(result_df["significant_bh"].sum())
    logger.info(
        "Granger: %d canonicals tested, %d significant (alpha=%.2f, BH-corrected)",
        int(tested_mask.sum()),
        n_sig,
        alpha,
    )
    return result_df
