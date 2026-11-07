"""Per-store statistical models on a store sample: Prophet and dynamic harmonic regression.

Same test origins and horizon as src.backtest. Both models fit log sales on trading days
with the same regressors: public holiday, retail event, the week before / after Christmas,
planned closures the day before / after, and weather (temperature vs local norm,
precipitation; observed = perfect forecast).
  Prophet: built-in weekly and yearly seasonality, independent errors.
  DHR:     day-of-week dummies + 4 yearly Fourier pairs as regressors, ARIMA errors chosen
           by AutoARIMA (statsforecast). ARIMA needs a gap-free series, so closed and
           anomalous days are linearly interpolated in log space for training.

Sample: every Longo's and Moto store, and 100 random Costa stores trading through the whole
test period. Fits run in 4 parallel processes.
Output: outputs/forecasts/stats_<client>.parquet, outputs/forecasts/stat_sample.csv
Run: python -m src.stat_models [client ...]
"""
import logging
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.backtest import OUT, test_origins
from src.features import HORIZON, MIN_HISTORY, precompute
from src.panel import load_panel

COSTA_SAMPLE = 100
SEED = 42
REGRESSORS = ["public", "event", "pre_xmas", "post_xmas", "closed_prev", "closed_next", "temp_anom", "precip_mm"]


def regressor_matrix(p, pre) -> np.ndarray:
    """(stores, days, regressors)."""
    n_s, n_d = p.sales.shape
    xmas = np.broadcast_to(pre["xmas_dist"], (n_s, n_d))
    closed = p.closed.astype(float)
    cols = [
        (p.public > 0).astype(float),
        (p.event > 0).astype(float),
        ((xmas >= -7) & (xmas <= -1)).astype(float),
        ((xmas >= 1) & (xmas <= 7)).astype(float),
        np.pad(closed, ((0, 0), (1, 0)))[:, :n_d],
        np.pad(closed, ((0, 0), (0, 1)))[:, 1:],
        np.nan_to_num(pre["temp_anom"]),
        np.nan_to_num(p.weather["precip_mm"]),
    ]
    return np.stack(cols, axis=2)


def dhr_terms(dates: pd.DatetimeIndex) -> np.ndarray:
    t = (dates - pd.Timestamp("2022-01-01")).days.to_numpy()
    dow = np.stack([(dates.dayofweek == k).astype(float) for k in range(6)], axis=1)
    fourier = [f(2 * np.pi * k * t / 365.25) for k in range(1, 5) for f in (np.sin, np.cos)]
    return np.column_stack([dow, *fourier])


def fit_store(task: dict) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    from prophet import Prophet
    from statsforecast.models import AutoARIMA

    dates, y, usable, X, fc = task["dates"], task["y"], task["usable"], task["X"], task["fc"]
    train_dates, future_dates = dates[:fc], dates[fc:]
    log_y = np.log(np.where(usable, y, np.nan))

    # Prophet: fits on usable days only (it tolerates gaps).
    df = pd.DataFrame(X[:fc], columns=REGRESSORS).assign(ds=train_dates, y=log_y[:fc])[usable[:fc]]
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False, uncertainty_samples=0)
    for r in REGRESSORS:
        m.add_regressor(r, standardize=True)
    m.fit(df)
    fut = pd.DataFrame(X[fc:], columns=REGRESSORS).assign(ds=future_dates)
    prophet = np.exp(m.predict(fut).yhat.to_numpy())

    # DHR: gap-free log series, calendar terms + regressors, ARIMA errors.
    filled = pd.Series(log_y[:fc]).interpolate(limit_direction="both").to_numpy()
    exog = np.column_stack([dhr_terms(dates), X])
    keep = exog[:fc].std(axis=0) > 0  # drop regressors constant in training (e.g. no snow)
    arima = AutoARIMA(season_length=1, max_p=3, max_q=3, max_d=1)
    arima.fit(filled, X=exog[:fc][:, keep])
    dhr = np.exp(arima.predict(len(future_dates), X=exog[fc:][:, keep])["mean"])

    return pd.DataFrame({"store_id": task["store_id"], "fold": task["fold"], "origin": train_dates[-1],
                         "date": future_dates, "pred_prophet": prophet, "pred_dhr": dhr})


def sample_stores(p) -> np.ndarray:
    first = test_origins(len(p.dates))[0]
    start = np.argmax(p.in_period, axis=1)
    eligible = np.flatnonzero(p.in_period[:, first] & p.in_period[:, -1] & (first - start >= MIN_HISTORY))
    if p.client != "costa":
        return eligible
    return np.sort(np.random.default_rng(SEED).choice(eligible, COSTA_SAMPLE, replace=False))


def run(client: str) -> None:
    t0 = time.time()
    p = load_panel(client)
    pre = precompute(p)
    X = regressor_matrix(p, pre)
    stores = sample_stores(p)
    tasks = []
    for k, o in enumerate(test_origins(len(p.dates)), 1):
        for s in stores:
            start = int(np.argmax(p.in_period[s]))
            end = min(o + HORIZON, len(p.dates) - 1)
            if not p.in_period[s, o] or o - start < MIN_HISTORY:
                continue
            sl = slice(start, end + 1)
            tasks.append({"store_id": p.stores[s], "fold": k, "dates": p.dates[sl], "y": p.sales[s, sl],
                          "usable": p.trading[s, sl], "X": X[s, sl], "fc": o - start + 1})
    print(f"[{client}] {len(stores)} stores, {len(tasks)} fits", flush=True)
    results = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(fit_store, tasks), 1):
            results.append(r)
            if i % 50 == 0 or i == len(tasks):
                print(f"[{client}] {i}/{len(tasks)} fits ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(results).to_parquet(OUT / f"stats_{client}.parquet", index=False)
    sample = pd.DataFrame({"client": client, "store_id": p.stores[stores]})
    path = OUT / "stat_sample.csv"
    if path.exists():
        sample = pd.concat([pd.read_csv(path).query("client != @client"), sample])
    sample.to_csv(path, index=False)
    print(f"[{client}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:] or ["longos", "moto", "costa"]:
        run(c)
