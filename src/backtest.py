"""Rolling-origin backtest: baselines and LightGBM on every store.

Per client: 6 test origins, 28 days apart, covering the last 168 days of data. Each origin
forecasts the next 28 days. Models are retrained per origin on rows whose target day is on
or before that origin (expanding window). LightGBM is one model per client across all its
stores (a separate model per client keeps each client's test period free of other clients'
later data).

Variants: lgbm (all features), lgbm_noweather, lgbm_noholiday (ablations).
Output: outputs/forecasts/<client>.parquet, outputs/forecasts/importance_<client>.csv
Run: python -m src.backtest [client ...]
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import INTERIM, ROOT
from src.features import CATEGORICAL, FEATURES, HOLIDAY_FEATURES, MIN_HISTORY, WEATHER_FEATURES, build_rows, precompute
from src.panel import load_panel

OUT = ROOT / "outputs" / "forecasts"
N_FOLDS = 6
FOLD_STEP = 28
PARAMS = {
    "objective": "l1",            # absolute error, weighted by store level below = WAPE
    "learning_rate": 0.1,
    "max_bin": 127,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 4,
    "verbose": -1,
    "seed": 7,
}
ROUNDS = 300
VARIANTS = {
    "lgbm": FEATURES,
    "lgbm_noweather": [f for f in FEATURES if f not in WEATHER_FEATURES],
    "lgbm_noholiday": [f for f in FEATURES if f not in HOLIDAY_FEATURES],
}


def test_origins(n_days: int) -> list[int]:
    end = n_days - 1
    return [end - FOLD_STEP * k for k in range(N_FOLDS, 0, -1)]


def feature_rows(client: str):
    """All weekly origins up to the last test origin (cached: building takes a while)."""
    p = load_panel(client)
    cache = INTERIM / f"features_{client}.parquet"
    if cache.exists():
        return p, pd.read_parquet(cache)
    last = test_origins(len(p.dates))[-1]
    origins = list(range(last, MIN_HISTORY - 1, -7))[::-1]
    rows = build_rows(p, origins, precompute(p))
    rows.to_parquet(cache, index=False)
    return p, rows


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    cats = [c for c in CATEGORICAL if c in features]
    ds = lgb.Dataset(train[features], train.y, weight=train.level, categorical_feature=cats, free_raw_data=True)
    model = lgb.train(PARAMS, ds, ROUNDS)
    return model.predict(test[features]) * test.level.to_numpy(), model


def run(client: str) -> None:
    t0 = time.time()
    p, rows = feature_rows(client)
    print(f"[{client}] {len(rows):,} feature rows ({time.time() - t0:.0f}s)", flush=True)
    usable = rows.trading & rows.y.notna()
    results, importance = [], None
    for k, o in enumerate(test_origins(len(p.dates)), 1):
        origin = p.dates[o]
        train = rows[usable & (rows.origin < origin) & (rows.date <= origin)]
        test = rows[(rows.origin == origin)].copy()
        for name, feats in VARIANTS.items():
            test[f"pred_{name}"], model = fit_predict(train, test, feats)
            if name == "lgbm" and k == N_FOLDS:
                importance = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
        test["fold"] = k
        results.append(test)
        print(f"[{client}] fold {k} origin {origin.date()}: train {len(train):,} rows, "
              f"test {len(test):,} ({time.time() - t0:.0f}s)", flush=True)

    keep = ["store_id", "fold", "origin", "date", "h", "actual", "trading", "level",
            "pred_naive", "pred_ma", "pred_toha", *[f"pred_{v}" for v in VARIANTS]]
    out = pd.concat(results)[keep]
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / f"{client}.parquet", index=False)
    (importance / importance.sum()).round(4).to_csv(OUT / f"importance_{client}.csv", header=["gain_share"])
    print(f"[{client}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:] or ["longos", "moto", "costa"]:
        run(c)
