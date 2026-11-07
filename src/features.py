"""Origin-based rows for direct multi-horizon forecasting, plus the baseline forecasts.

A row is (store, origin o, horizon h) with target day t = o + h. Every history feature uses
only days <= o, so a model trained on these rows never sees the future. History is scaled
by the store's level (mean trading-day sales over the 28 days to o), and the target is
sales[t] / level, which lets one model serve stores of any size.

Calendar, holiday and planned-closure features describe day t itself; they are known in
advance. Weather on day t is the observed value, i.e. a perfect forecast, so results with
weather are an upper bound on its value (compare against the no-weather ablation).
"""
import numpy as np
import pandas as pd
from dateutil.easter import easter

from src.panel import Panel, rolling_nanmean, same_weekday_mean

HORIZON = 28
MIN_HISTORY = 56  # days of trading history a store needs at the origin

CATEGORICAL = ["store", "public", "event"]
WEATHER_FEATURES = ["temp_max_c", "temp_anom", "precip_mm", "snow_cm", "wind_gust_kmh"]
HOLIDAY_FEATURES = ["public", "event", "days_to_public", "days_since_public", "xmas_dist", "easter_dist"]


def _days_to_next(flag: np.ndarray, cap: int) -> np.ndarray:
    """Days until the next flagged day (0 on the day itself), capped."""
    n = flag.shape[1]
    out = np.full(flag.shape, cap, dtype=np.int16)
    nxt = np.full(flag.shape[0], np.iinfo(np.int32).max // 2)
    for d in range(n - 1, -1, -1):
        nxt = np.where(flag[:, d], d, nxt)
        out[:, d] = np.minimum(nxt - d, cap)
    return out


def _signed_distance(dates: pd.DatetimeIndex, anchors: list[pd.Timestamp], cap: int) -> np.ndarray:
    """Signed days from each date to the nearest anchor date, clipped to +-cap."""
    a = np.array(anchors, dtype="datetime64[D]")
    d = dates.values.astype("datetime64[D]")
    diff = (d[:, None] - a[None, :]).astype(int)
    nearest = diff[np.arange(len(d)), np.abs(diff).argmin(axis=1)]
    return np.clip(nearest, -cap, cap)


def precompute(p: Panel) -> dict:
    hist = p.clean_sales
    years = range(p.dates[0].year - 1, p.dates[-1].year + 2)
    temp = p.weather["temp_max_c"]
    # Local temperature norm for the time of year: centred 31-day mean at the store's location.
    trailing = rolling_nanmean(temp, 31, 15)
    norm = np.concatenate([trailing[:, 15:], np.full((temp.shape[0], 15), np.nan)], axis=1)
    public_flag = p.public > 0
    return {
        "hist": hist,
        "level": rolling_nanmean(hist, 28, 14),
        "recent7": rolling_nanmean(hist, 7, 4),
        "dow8": same_weekday_mean(hist, 8, 3),
        "dow4": same_weekday_mean(hist, 4, 2),
        "temp_anom": temp - norm,
        "days_to_public": _days_to_next(public_flag, 14),
        "days_since_public": _days_to_next(public_flag[:, ::-1], 14)[:, ::-1],
        "xmas_dist": _signed_distance(p.dates, [pd.Timestamp(y, 12, 25) for y in years], 30),
        "easter_dist": _signed_distance(p.dates, [pd.Timestamp(easter(y)) for y in years], 14),
        "first_day": np.argmax(p.in_period, axis=1),
    }


def build_rows(p: Panel, origins: list[int], pre: dict | None = None) -> pd.DataFrame:
    """Feature rows, target and baseline forecasts for every store, origin and horizon."""
    pre = pre or precompute(p)
    hist, level = pre["hist"], pre["level"]
    n_days = len(p.dates)
    frames = []
    for o in origins:
        ok = (p.in_period[:, o] & ~np.isnan(level[:, o]) & (o - pre["first_day"] >= MIN_HISTORY))
        s = np.flatnonzero(ok)
        for h in range(1, HORIZON + 1):
            t = o + h
            if t >= n_days:
                break
            keep = p.in_period[s, t]
            si = s[keep]
            if len(si) == 0:
                continue
            frames.append(_rows(p, pre, si, o, h))
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def _rows(p: Panel, pre: dict, s: np.ndarray, o: int, h: int) -> pd.DataFrame:
    hist, level = pre["hist"], pre["level"]
    t = o + h
    d1 = t - 7 * int(np.ceil(h / 7))  # latest same weekday at or before the origin
    lvl = level[s, o]
    ly = t - 364                      # same weekday, one year earlier
    lvl_ly = level[s, o - 364] if o - 364 >= 0 else np.full(len(s), np.nan)

    def at(m, d):
        return m[s, d] if 0 <= d < m.shape[1] else np.full(len(s), np.nan)

    ly_avg = np.nanmean(np.stack([at(hist, ly - 7), at(hist, ly), at(hist, ly + 7)]), axis=0) \
        if ly - 7 >= 0 else np.full(len(s), np.nan)
    growth = lvl / lvl_ly
    date = p.dates[t]

    # Baselines, in sales units. Seasonal naive: the latest same weekday (falling back to the
    # 4-week same-weekday mean when that day was closed). MA: Dayforce-style moving average
    # of the last 4 same weekdays. ToHA: last year's same-weekday average scaled by the
    # year-on-year change in level (reconstructed from the client's description).
    ma = at(pre["dow4"], d1)
    naive = np.where(np.isnan(at(hist, d1)), ma, at(hist, d1))
    toha = np.where(np.isnan(ly_avg * growth), ma, ly_avg * growth)

    row = {
        "store_id": p.stores[s], "store": s.astype(np.int32), "origin": p.dates[o], "date": date,
        "h": np.full(len(s), h, dtype=np.int16),
        "actual": hist[s, t], "trading": p.trading[s, t], "level": lvl,
        "y": hist[s, t] / lvl,
        # calendar of the target day
        "dow": np.full(len(s), date.dayofweek, dtype=np.int16),
        "dom": np.full(len(s), date.day, dtype=np.int16),
        "month": np.full(len(s), date.month, dtype=np.int16),
        "woy": np.full(len(s), date.isocalendar().week, dtype=np.int16),
        "doy": np.full(len(s), date.dayofyear, dtype=np.int16),
        "trend": np.full(len(s), t, dtype=np.int32),
        # holidays and planned closures (known in advance)
        "public": p.public[s, t], "event": p.event[s, t],
        "days_to_public": pre["days_to_public"][s, t], "days_since_public": pre["days_since_public"][s, t],
        "xmas_dist": np.full(len(s), pre["xmas_dist"][t], dtype=np.int16),
        "easter_dist": np.full(len(s), pre["easter_dist"][t], dtype=np.int16),
        "closed_prev": at(p.closed, t - 1).astype(np.int8),
        "closed_next": at(p.closed, t + 1).astype(np.int8) if t + 1 < len(p.dates) else np.zeros(len(s), np.int8),
        # history, scaled by level
        "lag_w1": at(hist, d1) / lvl, "lag_w2": at(hist, d1 - 7) / lvl,
        "lag_w3": at(hist, d1 - 14) / lvl, "lag_w4": at(hist, d1 - 21) / lvl,
        "dow8": at(pre["dow8"], d1) / lvl, "recent7": pre["recent7"][s, o] / lvl,
        "last_year": at(hist, ly) / lvl_ly, "last_year_avg": ly_avg / lvl_ly, "growth": growth,
        # weather on the target day (observed = perfect forecast)
        "temp_max_c": p.weather["temp_max_c"][s, t], "temp_anom": pre["temp_anom"][s, t],
        "precip_mm": p.weather["precip_mm"][s, t], "snow_cm": p.weather["snow_cm"][s, t],
        "wind_gust_kmh": p.weather["wind_gust_kmh"][s, t],
        # baselines
        "pred_naive": naive, "pred_ma": ma, "pred_toha": toha,
    }
    return pd.DataFrame(row)


FEATURES = [
    "store", "h", "dow", "dom", "month", "woy", "doy", "trend",
    "public", "event", "days_to_public", "days_since_public", "xmas_dist", "easter_dist",
    "closed_prev", "closed_next",
    "lag_w1", "lag_w2", "lag_w3", "lag_w4", "dow8", "recent7", "last_year", "last_year_avg", "growth",
    *WEATHER_FEATURES,
]
