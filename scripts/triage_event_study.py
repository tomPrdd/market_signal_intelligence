"""Triage: does anything survive a properly specified event study?

Recomputes Track B three ways and compares, WITHOUT touching the production
pipeline:
  1. mean-adjusted (what the pipeline does today)
  2. market model  (alpha + beta * CAC40, the standard specification)
  3. + Benjamini-Hochberg correction, + n_events threshold

Then runs the two decisive diagnostics:
  - permutation test: shuffle which dates belong to which mechanism, rebuild the
    whole result 1000x, and see whether the real result stands out from the null
  - polarity validity: do +1 mechanisms earn more than -1 mechanisms?

Key efficiency insight: a CAR depends only on the EVENT DATE, not on the
mechanism. There are ~100 distinct dates per company but thousands of
"events", so we compute each date's CAR once and reuse it. This also makes the
overlap problem explicit.

    uv run python scripts/triage_event_study.py ML.PA OR.PA SAN.PA MC.PA TTE.PA
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEMO = Path("app/demo")
MARKET = Path("data/market")
EST_START, EST_END = -120, -11
CAR_START, CAR_END = 1, 30
MIN_EST_DAYS = 30
INDEX = "^FCHI"  # CAC 40 — all five companies are Paris-listed


def _log_returns(prices: pd.DataFrame) -> pd.Series:
    col = "adj_close" if "adj_close" in prices.columns else "close"
    s = np.log(prices[col] / prices[col].shift(1)).dropna()
    s.index = pd.to_datetime(s.index)
    return s


def _index_returns() -> pd.Series:
    MARKET.mkdir(parents=True, exist_ok=True)
    cache = MARKET / "FCHI_prices.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col="date", parse_dates=True)
    else:
        raw = yf.download(INDEX, period="max", progress=False, auto_adjust=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower().replace(" ", "_") for c in raw.columns]
        else:
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        df = raw[[c for c in ["close", "adj_close"] if c in raw.columns]].copy()
        df.index.name = "date"
        df.to_csv(cache, index_label="date")
    return _log_returns(df)


def _car_for_dates(dates, r_stock: pd.Series, r_mkt: pd.Series) -> dict:
    """CAR per event date, under both specifications. {date: (mean_adj, market)}"""
    idx = r_stock.index.sort_values()
    out = {}
    for d in dates:
        ref = pd.Timestamp(d)
        loc = idx.searchsorted(ref, side="left")
        es, ee = loc + EST_START, loc + EST_END
        cs, ce = loc + CAR_START, loc + CAR_END
        if es < 0 or ce >= len(idx):
            continue
        est = r_stock.iloc[es : ee + 1]
        evt = r_stock.iloc[cs : ce + 1]
        if len(est) < MIN_EST_DAYS or evt.empty:
            continue

        car_mean = float((evt - est.mean()).sum())

        # market model: align stock & index over the estimation window
        est_m = r_mkt.reindex(est.index).dropna()
        common = est.index.intersection(est_m.index)
        car_mkt = np.nan
        if len(common) >= MIN_EST_DAYS:
            y, x = est.loc[common].values, est_m.loc[common].values
            var = x.var()
            if var > 0:
                beta = float(np.cov(y, x, ddof=1)[0, 1] / var)
                alpha = float(y.mean() - beta * x.mean())
                evt_m = r_mkt.reindex(evt.index)
                expected = alpha + beta * evt_m
                ar = (evt - expected).dropna()
                if len(ar) >= (CAR_END - CAR_START) * 0.7:
                    car_mkt = float(ar.sum())
        out[pd.Timestamp(d).normalize()] = (car_mean, car_mkt)
    return out


def _mechanism_stats(members: dict, car_by_date: dict, key: int) -> pd.DataFrame:
    """key=0 mean-adjusted, key=1 market model. One row per canonical."""
    rows = []
    for cid, (dates, polarity, presence) in members.items():
        vals = [car_by_date[d][key] for d in dates if d in car_by_date]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        n_dates = len({d for d in dates if d in car_by_date})
        if len(vals) < 2:
            continue
        arr = np.array(vals)
        sd = arr.std(ddof=1)
        t = arr.mean() / (sd / np.sqrt(len(arr))) if sd > 0 else np.nan
        p = float(2 * stats.t.sf(abs(t), df=len(arr) - 1)) if np.isfinite(t) else np.nan
        rows.append(
            {
                "canonical_id": cid,
                "n_events": len(arr),
                "n_dates": n_dates,
                "mean_car": arr.mean(),
                "p_value": p,
                "polarity": polarity,
                "corpus_presence": presence,
            }
        )
    return pd.DataFrame(rows)


def _bh(p: np.ndarray, alpha=0.05) -> int:
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return 0
    srt = np.sort(p)
    thresh = alpha * np.arange(1, len(srt) + 1) / len(srt)
    below = srt <= thresh
    return int(np.max(np.where(below)[0]) + 1) if below.any() else 0


def analyse(ticker: str, n_perm: int = 1000) -> dict:
    pool = DEMO / f"{ticker}_ptc_pool.jsonl"
    canon = pd.read_csv(DEMO / f"{ticker}_canonicals.csv")
    prices = pd.read_csv(MARKET / f"{ticker}_prices.csv", index_col="date", parse_dates=True)

    sid_date = {}
    for line in pool.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid_date[r["source_id"]] = pd.Timestamp(r["source_date"]).normalize()

    if "is_noise" in canon.columns:
        canon = canon[~canon["is_noise"].astype(str).str.lower().eq("true")]
    canon = canon[canon["direction"] == "precursor"]

    members = {}
    for _, c in canon.iterrows():
        try:
            sids = json.loads(c["source_ids"])
        except (TypeError, ValueError):
            continue
        dates = [sid_date[s] for s in sids if s in sid_date]
        if len(dates) >= 2:
            members[c["canonical_id"]] = (dates, int(c["polarity"]), c.get("corpus_presence", ""))
    if not members:
        return {"ticker": ticker, "error": "no testable mechanisms"}

    all_dates = sorted({d for v in members.values() for d in v[0]})
    r_stock, r_mkt = _log_returns(prices), _index_returns()
    car_by_date = _car_for_dates(all_dates, r_stock, r_mkt)

    res = {"ticker": ticker, "n_mechanisms": len(members), "n_distinct_dates": len(car_by_date)}

    for key, label in [(0, "mean_adj"), (1, "market")]:
        df = _mechanism_stats(members, car_by_date, key)
        if df.empty:
            continue
        strict = df[df.n_events >= 5]
        res[f"{label}_n"] = len(df)
        res[f"{label}_raw_sig"] = int((df.p_value < 0.05).sum())
        res[f"{label}_bh"] = _bh(df.p_value.values)
        res[f"{label}_bh_n5"] = _bh(strict.p_value.values) if len(strict) else 0
        res[f"{label}_mean_car"] = float(df.mean_car.mean())
        res[f"{label}_pct_pos"] = float((df.mean_car > 0).mean() * 100)
        pos, neg = df[df.polarity > 0].mean_car, df[df.polarity < 0].mean_car
        if len(pos) > 5 and len(neg) > 5:
            tt = stats.ttest_ind(pos, neg, equal_var=False)
            res[f"{label}_pol_spread"] = float(pos.mean() - neg.mean())
            res[f"{label}_pol_p"] = float(tt.pvalue)
        res[f"{label}_df"] = df

    # single-date artefacts: "several events" that are really one publication
    any_df = res.get("market_df", res.get("mean_adj_df"))
    if any_df is not None and not any_df.empty:
        res["single_date_pct"] = float((any_df.n_dates == 1).mean() * 100)

    # ---- DEMEANED specification: does a mechanism beat the AVERAGE document date? ----
    # Testing mean_car != 0 mostly rediscovers the stock's drift over the period:
    # if every event date has a positive CAR, every mechanism tests "significant".
    # The discriminating question is whether a mechanism differs from a random date.
    cars_all = np.array([v[1] for v in car_by_date.values()])
    grand_mean = float(np.nanmean(cars_all))
    res["grand_mean_car"] = grand_mean
    demeaned = {d: (v[0], v[1] - grand_mean) for d, v in car_by_date.items()}
    df_dm = _mechanism_stats(members, demeaned, 1)
    if not df_dm.empty:
        res["demeaned_n"] = len(df_dm)
        res["demeaned_raw_sig"] = int((df_dm.p_value < 0.05).sum())
        res["demeaned_bh"] = _bh(df_dm.p_value.values)
        strict = df_dm[df_dm.n_events >= 5]
        res["demeaned_bh_n5"] = _bh(strict.p_value.values) if len(strict) else 0

    # ---- permutation test, preserving each mechanism's DATE structure ----
    # A mechanism's events often repeat the same date (same document). Repeated
    # values shrink the within-mechanism variance and inflate |t|. The null must
    # reproduce that structure, so we reassign *which* dates a mechanism gets
    # while keeping how many distinct dates and their multiplicities.
    df_real = res.get("market_df")
    if df_real is not None and not df_real.empty:
        usable = [d for d, v in car_by_date.items() if np.isfinite(v[1])]
        car_lookup = {d: car_by_date[d][1] for d in usable}
        structures = []
        for dates, pol, _pres in members.values():
            counts = pd.Series([d for d in dates if d in car_lookup]).value_counts()
            if counts.sum() >= 2:
                structures.append((list(counts.values), pol))
        rng = np.random.default_rng(42)
        pool_dates = np.array(usable, dtype=object)
        null_bh, null_spread = [], []
        for _ in range(n_perm):
            ps, means, pols = [], [], []
            for mult, pol in structures:
                k = min(len(mult), len(pool_dates))
                picks = rng.choice(len(pool_dates), size=k, replace=False)
                vals = []
                for m, pi in zip(mult[:k], picks, strict=False):
                    vals.extend([car_lookup[pool_dates[pi]]] * int(m))
                draw = np.array(vals)
                if len(draw) < 2:
                    continue
                sd = draw.std(ddof=1)
                if sd > 0:
                    t = draw.mean() / (sd / np.sqrt(len(draw)))
                    ps.append(float(2 * stats.t.sf(abs(t), df=len(draw) - 1)))
                    means.append(draw.mean())
                    pols.append(pol)
            if ps:
                null_bh.append(_bh(np.array(ps)))
                m_, p_ = np.array(means), np.array(pols)
                if (p_ > 0).sum() > 5 and (p_ < 0).sum() > 5:
                    null_spread.append(m_[p_ > 0].mean() - m_[p_ < 0].mean())
        if null_bh:
            res["null_bh_median"] = float(np.median(null_bh))
            res["null_bh_p95"] = float(np.percentile(null_bh, 95))
            real_bh = res.get("market_bh", 0)
            res["perm_p"] = float(np.mean([b >= real_bh for b in null_bh]))
        if null_spread and "market_pol_spread" in res:
            obs = abs(res["market_pol_spread"])
            res["perm_pol_p"] = float(np.mean([abs(s) >= obs for s in null_spread]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--perm", type=int, default=1000)
    args = ap.parse_args()

    all_res, frames = [], []
    for t in args.tickers:
        try:
            r = analyse(t, n_perm=args.perm)
        except Exception as e:
            print(f"{t}: ERREUR {e}")
            continue
        if "error" in r:
            print(f"{t}: {r['error']}")
            continue
        all_res.append(r)
        d = r.get("market_df")
        if d is not None:
            d = d.copy()
            d["ticker"] = t
            frames.append(d)

    print("\n" + "=" * 78)
    print("A. SPÉCIFICATION ACTUELLE (mean-adjusted)  vs  MODÈLE DE MARCHÉ")
    print("=" * 78)
    print(
        f"{'':10} {'meca':>5} {'dates':>6} | {'brut':>5} {'BH':>4} | "
        f"{'brut':>5} {'BH':>4} | {'brut':>5} {'BH':>4} {'BH+n5':>6}"
    )
    print(
        f"{'':10} {'':>5} {'':>6} | mean-adjust | modele marche | marche DEMEANE (vs date moyenne)"
    )
    for r in all_res:
        print(
            f"{r['ticker']:10} {r['n_mechanisms']:5} {r['n_distinct_dates']:6} | "
            f"{r.get('mean_adj_raw_sig', 0):5} {r.get('mean_adj_bh', 0):4} | "
            f"{r.get('market_raw_sig', 0):5} {r.get('market_bh', 0):4} | "
            f"{r.get('demeaned_raw_sig', 0):5} {r.get('demeaned_bh', 0):4} {r.get('demeaned_bh_n5', 0):6}"
        )

    print("\n" + "=" * 78)
    print("B. BIAIS DIRECTIONNEL (le modèle de marché doit ramener % positifs vers 50%)")
    print("=" * 78)
    for r in all_res:
        print(
            f"  {r['ticker']:10} CAR moyen {r.get('mean_adj_mean_car', 0):+.2%} -> "
            f"{r.get('market_mean_car', 0):+.2%}   |   % positifs "
            f"{r.get('mean_adj_pct_pos', 0):.0f}% -> {r.get('market_pct_pos', 0):.0f}%"
            f"   |   'événements' sur 1 seule date: {r.get('single_date_pct', 0):.0f}%"
        )

    print("\n" + "=" * 78)
    print("C. TEST DE PERMUTATION (le vrai résultat sort-il du hasard ?)")
    print("=" * 78)
    for r in all_res:
        print(
            f"  {r['ticker']:10} BH réel = {r.get('market_bh', 0):3}  |  "
            f"hasard: médiane {r.get('null_bh_median', float('nan')):5.1f}, "
            f"95e pct {r.get('null_bh_p95', float('nan')):5.1f}  |  p_perm = "
            f"{r.get('perm_p', float('nan')):.3f}"
        )

    print("\n" + "=" * 78)
    print("D. TEST DE VALIDITÉ: la polarité prédit-elle le CAR ? (modèle de marché)")
    print("=" * 78)
    for r in all_res:
        if "market_pol_spread" in r:
            print(
                f"  {r['ticker']:10} ecart(+1 vs -1) = {r['market_pol_spread']:+.2%}  "
                f"p={r['market_pol_p']:.3f}  |  p_permutation={r.get('perm_pol_p', float('nan')):.3f}"
            )
    if frames:
        P = pd.concat(frames)
        pos, neg = P[P.polarity > 0].mean_car, P[P.polarity < 0].mean_car
        tt = stats.ttest_ind(pos, neg, equal_var=False)
        print(
            f"\n  POOLED: +1 = {pos.mean():+.2%} (n={len(pos)}) | "
            f"-1 = {neg.mean():+.2%} (n={len(neg)})"
        )
        print(f"  écart = {pos.mean() - neg.mean():+.2%}  t={tt.statistic:.2f}  p={tt.pvalue:.4f}")
        verdict = "SIGNAL" if (tt.pvalue < 0.05 and pos.mean() > neg.mean()) else "PAS DE SIGNAL"
        print(f"  --> {verdict}")


if __name__ == "__main__":
    main()
