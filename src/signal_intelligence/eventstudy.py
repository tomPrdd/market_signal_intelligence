"""Phase 7B — Event study: cumulative abnormal returns per canonical.

For each PTC tagged as a precursor, compute the cumulative abnormal return
(CAR) over [+1, +30] trading days relative to source_date. Aggregate per
canonical mechanism to get an abnormal-return profile.

Methodology
-----------
- Normal return estimated as the mean daily return over the estimation window
  [-120, -11] trading days before the event date.
- Abnormal return for day t = actual_return(t) - normal_return.
- CAR = sum of daily abnormal returns over [+1, +30].
- Per-canonical aggregation: mean CAR, std, count, t-stat.

This is Track B — it has real statistical power because N is O(trading days),
not O(quarters).

Output schema (one row per canonical)
--------------------------------------
canonical_id        str
direction           str
n_events            int    (number of PTCs / source_dates contributing)
mean_car            float  (mean CAR over [+1, +30])
std_car             float
t_stat              float  (mean_car / (std_car / sqrt(n_events)))
p_value             float  (two-tailed t-test p-value)
positive_pct        float  (fraction of events with CAR > 0)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_ESTIMATION_START = -120  # trading days before event
_ESTIMATION_END = -11  # trading days before event (inclusive)
_CAR_START = 1  # first day of event window
_CAR_END = 30  # last day of event window (inclusive)
_MIN_ESTIMATION_DAYS = 30  # minimum days required to estimate normal return


def _daily_returns(prices: pd.DataFrame) -> pd.Series:
    """Compute daily log returns from adj_close. Index = date."""
    col = "adj_close" if "adj_close" in prices.columns else "close"
    return np.log(prices[col] / prices[col].shift(1)).dropna()


def _trading_day_offset(date_index: pd.DatetimeIndex, ref_date, offset: int):
    """Return the trading date at `offset` trading days from ref_date.

    Returns None if out of bounds.
    """
    sorted_idx = date_index.sort_values()
    loc_arr = sorted_idx.searchsorted(ref_date, side="left")
    target = loc_arr + offset
    if target < 0 or target >= len(sorted_idx):
        return None
    return sorted_idx[target]


def compute_car_path(
    event_date,
    returns: pd.Series,
    estimation_start: int = _ESTIMATION_START,
    estimation_end: int = _ESTIMATION_END,
    car_start: int = _CAR_START,
    car_end: int = _CAR_END,
) -> pd.Series | None:
    """Compute the cumulative abnormal return path for a single event date.

    Returns a Series indexed by relative trading day (car_start..car_end),
    or None if there are insufficient data points.
    """
    date_idx = pd.DatetimeIndex(returns.index)
    ref = pd.Timestamp(event_date)

    est_s = _trading_day_offset(date_idx, ref, estimation_start)
    est_e = _trading_day_offset(date_idx, ref, estimation_end)
    car_s = _trading_day_offset(date_idx, ref, car_start)
    car_e = _trading_day_offset(date_idx, ref, car_end)

    if any(d is None for d in [est_s, est_e, car_s, car_e]):
        return None

    estimation_window = returns.loc[est_s:est_e]
    if len(estimation_window) < _MIN_ESTIMATION_DAYS:
        return None

    normal_return = float(estimation_window.mean())
    event_window = returns.loc[car_s:car_e]
    if event_window.empty:
        return None

    path = (event_window - normal_return).cumsum()
    path.index = range(car_start, car_start + len(path))
    return path


def compute_car(
    event_date,
    returns: pd.Series,
    estimation_start: int = _ESTIMATION_START,
    estimation_end: int = _ESTIMATION_END,
    car_start: int = _CAR_START,
    car_end: int = _CAR_END,
) -> float | None:
    """Compute CAR for a single event date.

    Returns None if there are insufficient data points.
    """
    path = compute_car_path(
        event_date, returns, estimation_start, estimation_end, car_start, car_end
    )
    return None if path is None else float(path.iloc[-1])


def run_event_study(
    canonical_df: pd.DataFrame,
    pool_path: Path,
    prices_df: pd.DataFrame,
    output_dir: Path | None = None,
    ticker: str = "",
) -> pd.DataFrame:
    """Run the full Track B event study for all precursor canonicals.

    Parameters
    ----------
    canonical_df:
        Output of cluster_ptcs(). Used to map canonical → source_ids and direction.
    pool_path:
        JSONL PTC pool. Used to look up source_date per source_id.
    prices_df:
        Output of get_prices(). Must contain adj_close or close column.
    output_dir:
        If provided, saves results CSV to output_dir/{ticker}_eventstudy.csv.
    ticker:
        Used for the output filename and logging.

    Returns
    -------
    DataFrame with one row per canonical, schema as described in module docstring.
    Canonicals with direction == "consequence" are still included with n_events=0
    to provide a complete inventory.
    """
    import json

    from ptc_extraction.io import load_pool

    ptcs = load_pool(pool_path)
    sid_to_date: dict[str, pd.Timestamp] = {p.source_id: pd.Timestamp(p.source_date) for p in ptcs}

    prices_df = prices_df.copy()
    prices_df.index = pd.to_datetime(prices_df.index)
    returns = _daily_returns(prices_df)

    rows: list[dict] = []

    for _, canon_row in canonical_df.iterrows():
        cid = canon_row["canonical_id"]
        direction = canon_row["direction"]

        try:
            source_ids: list[str] = json.loads(canon_row["source_ids"])
        except (TypeError, ValueError):
            source_ids = []

        if direction != "precursor" or not source_ids:
            rows.append(
                {
                    "canonical_id": cid,
                    "direction": direction,
                    "n_events": 0,
                    "mean_car": None,
                    "std_car": None,
                    "t_stat": None,
                    "p_value": None,
                    "positive_pct": None,
                }
            )
            continue

        cars: list[float] = []
        for sid in source_ids:
            event_date = sid_to_date.get(sid)
            if event_date is None:
                continue
            car = compute_car(event_date, returns)
            if car is not None:
                cars.append(car)

        if len(cars) < 2:
            rows.append(
                {
                    "canonical_id": cid,
                    "direction": direction,
                    "n_events": len(cars),
                    "mean_car": float(cars[0]) if cars else None,
                    "std_car": None,
                    "t_stat": None,
                    "p_value": None,
                    "positive_pct": float(cars[0] > 0) if cars else None,
                }
            )
            continue

        cars_arr = np.array(cars)
        mean_car = float(cars_arr.mean())
        std_car = float(cars_arr.std(ddof=1))
        t_stat = mean_car / (std_car / np.sqrt(len(cars_arr))) if std_car > 0 else 0.0
        p_value = float(2 * stats.t.sf(abs(t_stat), df=len(cars_arr) - 1))
        positive_pct = float((cars_arr > 0).mean() * 100)

        rows.append(
            {
                "canonical_id": cid,
                "direction": direction,
                "n_events": len(cars_arr),
                "mean_car": round(mean_car, 6),
                "std_car": round(std_car, 6),
                "t_stat": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "positive_pct": round(positive_pct, 1),
            }
        )

    result_df = pd.DataFrame(rows).sort_values("canonical_id").reset_index(drop=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{ticker}_eventstudy.csv"
        result_df.to_csv(out_path, index=False)
        logger.info("Saved event study results to %s (%d rows)", out_path, len(result_df))

    tested = result_df[result_df["n_events"] >= 2]
    logger.info(
        "Event study: %d precursor canonicals with >=2 events, mean CAR=%.4f",
        len(tested),
        float(tested["mean_car"].mean()) if not tested.empty else 0.0,
    )
    return result_df


def run_car_profiles(
    canonical_df: pd.DataFrame,
    pool_path: Path,
    prices_df: pd.DataFrame,
    output_dir: Path | None = None,
    ticker: str = "",
) -> pd.DataFrame:
    """Compute the mean daily CAR trajectory per precursor canonical.

    Long format, one row per (canonical_id, rel_day):

    canonical_id   str
    rel_day        int    (1..30 trading days after source_date)
    mean_car       float  (mean cumulative abnormal return across events)
    sem_car        float  (standard error of the mean; None when n_events < 2)
    n_events       int

    Only precursor canonicals with at least one full-window event are included.
    Saves to output_dir/{ticker}_car_profiles.csv when output_dir is given.
    """
    import json

    from ptc_extraction.io import load_pool

    ptcs = load_pool(pool_path)
    sid_to_date: dict[str, pd.Timestamp] = {p.source_id: pd.Timestamp(p.source_date) for p in ptcs}

    prices_df = prices_df.copy()
    prices_df.index = pd.to_datetime(prices_df.index)
    returns = _daily_returns(prices_df)

    rows: list[dict] = []
    for _, canon_row in canonical_df.iterrows():
        if canon_row["direction"] != "precursor":
            continue
        try:
            source_ids: list[str] = json.loads(canon_row["source_ids"])
        except (TypeError, ValueError):
            continue

        paths: list[np.ndarray] = []
        for sid in source_ids:
            event_date = sid_to_date.get(sid)
            if event_date is None:
                continue
            path = compute_car_path(event_date, returns)
            if path is not None and len(path) == _CAR_END - _CAR_START + 1:
                paths.append(path.to_numpy())

        if not paths:
            continue

        matrix = np.vstack(paths)
        n_events = matrix.shape[0]
        mean_path = matrix.mean(axis=0)
        sem_path = matrix.std(axis=0, ddof=1) / np.sqrt(n_events) if n_events >= 2 else None

        for i, rel_day in enumerate(range(_CAR_START, _CAR_END + 1)):
            rows.append(
                {
                    "canonical_id": canon_row["canonical_id"],
                    "rel_day": rel_day,
                    "mean_car": round(float(mean_path[i]), 6),
                    "sem_car": round(float(sem_path[i]), 6) if sem_path is not None else None,
                    "n_events": n_events,
                }
            )

    result_df = pd.DataFrame(
        rows, columns=["canonical_id", "rel_day", "mean_car", "sem_car", "n_events"]
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{ticker}_car_profiles.csv"
        result_df.to_csv(out_path, index=False)
        logger.info("Saved CAR profiles to %s (%d rows)", out_path, len(result_df))

    return result_df
