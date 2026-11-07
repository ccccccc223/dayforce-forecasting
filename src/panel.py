"""Store x day matrices for one client, the basis for features, baselines and backtests.

Every matrix has shape (n_stores, n_days) over one shared daily calendar; days outside a
store's trading period are NaN (sales) or 0 (flags).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import PROCESSED

WEATHER = ["temp_max_c", "temp_min_c", "precip_mm", "snow_cm", "wind_speed_kmh", "wind_gust_kmh"]


@dataclass
class Panel:
    client: str
    stores: np.ndarray          # store ids, row order
    dates: pd.DatetimeIndex     # column order
    sales: np.ndarray           # raw sales, NaN outside trading period
    trading: np.ndarray         # bool: open, not an anomaly -> usable as target / history
    closed: np.ndarray          # bool: planned closure
    in_period: np.ndarray       # bool: inside the store's trading period
    public: np.ndarray          # int code of public holiday (0 = none), per store region
    event: np.ndarray           # int code of retail event (0 = none)
    weather: dict = field(default_factory=dict)  # name -> matrix
    holiday_names: list = field(default_factory=list)
    event_names: list = field(default_factory=list)

    def day(self, ts) -> int:
        return int(self.dates.get_loc(pd.Timestamp(ts)))

    @property
    def clean_sales(self) -> np.ndarray:
        """Sales on usable trading days only (NaN otherwise): the history models may learn from."""
        return np.where(self.trading, self.sales, np.nan)


def _name(s: pd.Series) -> pd.Series:
    """One label per day: substitute days are their own category, multi-name days keep the first."""
    first = s.str.split(r" \| |; ").str[0]
    return first.where(~s.str.contains("observed", na=False), "Substitute holiday")


def load_panel(client: str) -> Panel:
    df = pd.read_parquet(PROCESSED / "analysis_daily.parquet")
    df = df[df.client == client]
    stores = np.sort(df.store_id.unique())
    dates = pd.date_range(df.date.min(), df.date.max(), freq="D")

    def matrix(col, fill=np.nan, dtype=float):
        m = df.pivot(index="store_id", columns="date", values=col).reindex(index=stores, columns=dates)
        return m.fillna(fill).to_numpy(dtype=dtype) if fill is not np.nan else m.to_numpy(dtype=dtype)

    in_period = ~np.isnan(matrix("sales"))
    sales = matrix("sales")
    closed = matrix("is_closed", 0).astype(bool)
    anomaly = matrix("is_anomaly", 0).astype(bool)

    # Holiday codes come from the region calendar so they exist for every day, not only
    # days inside a store's trading period.
    cal = pd.read_parquet(PROCESSED / "holidays.parquet")
    cal = cal[cal.date.between(dates[0], dates[-1])]
    regions = df.drop_duplicates("store_id").set_index("store_id").region.reindex(stores)
    hol_names = sorted(_name(cal[cal.kind == "public"].name).unique())
    ev_names = sorted(cal[cal.kind == "event"].name.unique())
    public = np.zeros((len(stores), len(dates)), dtype=np.int16)
    event = np.zeros_like(public)
    for kind, names, target in (("public", hol_names, public), ("event", ev_names, event)):
        c = cal[cal.kind == kind].assign(label=lambda x: _name(x.name) if kind == "public" else x.name)
        c = c.drop_duplicates(["region", "date"])
        for i, region in enumerate(regions):
            r = c[c.region == region]
            target[i, dates.get_indexer(r.date)] = [names.index(n) + 1 for n in r.label]

    weather = {w: matrix(w) for w in WEATHER}
    return Panel(
        client=client, stores=stores, dates=dates, sales=sales,
        trading=in_period & ~closed & ~anomaly, closed=closed, in_period=in_period,
        public=public, event=event, weather=weather, holiday_names=hol_names, event_names=ev_names,
    )


def rolling_nanmean(m: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """Trailing mean over `window` days ending at each day (inclusive), ignoring NaN."""
    valid = ~np.isnan(m)
    csum = np.cumsum(np.where(valid, m, 0.0), axis=1)
    ccnt = np.cumsum(valid, axis=1)
    pad = lambda a: np.concatenate([np.zeros((a.shape[0], window)), a], axis=1)  # noqa: E731
    s = pad(csum)[:, window:] - pad(csum)[:, :-window]
    c = pad(ccnt)[:, window:] - pad(ccnt)[:, :-window]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c >= min_count, s / c, np.nan)


def same_weekday_mean(m: np.ndarray, weeks: int, min_count: int) -> np.ndarray:
    """Mean of the same weekday over the last `weeks` weeks ending at each day (inclusive)."""
    stack = np.stack([np.pad(m, ((0, 0), (7 * k, 0)), constant_values=np.nan)[:, :m.shape[1]]
                      for k in range(weeks)])
    cnt = (~np.isnan(stack)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        return np.where(cnt >= min_count, np.nanmean(stack, axis=0), np.nan)
