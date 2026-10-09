"""Export per-mechanism robustness columns into the demo bundle.

Reuses the validated triage analysis to produce app/demo/{ticker}_robustness.csv:

    canonical_id, n_events, n_dates, mean_car_market, p_value_market, p_bh_market

- n_dates          : distinct publication dates behind the "events" (the real
                     number of independent observations; often far below n_events)
- mean_car_market  : CAR under the standard market model (alpha + beta*CAC40)
                     rather than the naive mean-adjusted specification
- p_bh_market      : Benjamini-Hochberg corrected p-value across all mechanisms

The app shows these next to the raw numbers so a reader can see how much of the
headline result survives a correct specification.

    uv run python scripts/export_robustness.py ML.PA OR.PA SAN.PA MC.PA TTE.PA
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triage_event_study import analyse

DEMO = Path("app/demo")


def _bh_pvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (same length/order as input)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    vals = p[ok]
    n = len(vals)
    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    adj = np.clip(adj, 0, 1)
    restored = np.empty(n)
    restored[order] = adj
    out[ok] = restored
    return out


def export(ticker: str) -> None:
    res = analyse(ticker, n_perm=0)
    df = res.get("market_df")
    if df is None or df.empty:
        print(f"{ticker}: aucun mécanisme testable, ignoré")
        return
    out = df[["canonical_id", "n_events", "n_dates", "mean_car", "p_value"]].copy()
    out = out.rename(columns={"mean_car": "mean_car_market", "p_value": "p_value_market"})
    out["p_bh_market"] = _bh_pvalues(out["p_value_market"].values)
    dest = DEMO / f"{ticker}_robustness.csv"
    out.to_csv(dest, index=False)
    n_sig = int((out.p_bh_market < 0.05).sum())
    print(
        f"{ticker}: {len(out)} mécanismes | {res['n_distinct_dates']} dates distinctes | "
        f"{n_sig} significatifs après BH -> {dest.name}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tickers", nargs="+")
    args = ap.parse_args()
    for t in args.tickers:
        try:
            export(t)
        except Exception as e:
            print(f"{t}: ERREUR {e}")


if __name__ == "__main__":
    main()
