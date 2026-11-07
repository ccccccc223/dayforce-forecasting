"""Exploratory analysis of the merged daily table. Writes figures and the numbers behind them.

All sales are shown as an index relative to each store's own median trading-day sales,
so stores of different size are comparable and no absolute figures are published.

  idx   sales / store median                    (level-free sales)
  lift  sales / same-weekday baseline           (deviation from a normal day: the median
        of the same weekday 1-3 weeks before and after, ignoring holidays and closures)

Output: reports/figures/*.png, reports/eda_stats.json
Run: python -m src.eda
"""
import json

import numpy as np
import pandas as pd

from src.config import PROCESSED, REPORTS
from src.plotstyle import CLIENT_COLORS, CLIENT_LABELS, CLIENTS, INK_2, MUTED, label_end, plt

FIG = REPORTS / "figures"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
stats: dict = {}


def load() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "analysis_daily.parquet").sort_values(["client", "store_id", "date"])
    df = df.reset_index(drop=True)
    store = [df.client, df.store_id]
    df["trading"] = (df.is_closed == 0) & (df.is_anomaly == 0)
    df["holiday"] = df.public_holiday.notna() | df.retail_event.notna() | df.client_special_day.notna()

    median = df.sales.where(df.trading).groupby(store).transform("median")
    df["idx"] = df.sales.where(df.is_anomaly == 0) / median

    # Every store has a complete daily calendar, so a shift of 7 rows is the same weekday.
    normal = df.sales.where(df.trading & ~df.holiday).groupby(store)
    neighbours = pd.concat([normal.shift(k) for k in (7, 14, 21, -7, -14, -21)], axis=1)
    baseline = neighbours.median(axis=1).where(neighbours.notna().sum(axis=1) >= 3)
    df["lift"] = df.sales.where(df.is_anomaly == 0) / baseline
    return df


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def trend(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for c in CLIENTS:
        d = df[(df.client == c) & df.trading].groupby("date").idx.mean().rolling(28, center=True).mean()
        ax.plot(d.index, d.values, color=CLIENT_COLORS[c], label=CLIENT_LABELS[c])
        label_end(ax, d.dropna().index[-1], d.dropna().iloc[-1], c)
    ax.set_title("Sales level over time (28-day average, index: store median = 1)")
    ax.set_ylabel("Sales index")
    ax.legend(loc="upper left", ncols=3)
    save(fig, "01_trend")

    # Year-on-year change for the months every year shares (Jan-Aug), same stores only.
    yoy = {}
    for c in CLIENTS:
        d = df[(df.client == c) & df.trading & (df.month <= 8)]
        by = d.groupby(["store_id", "year"]).idx.mean().unstack()
        yoy[c] = {f"{a}->{b}": round(float((by[b] / by[a]).dropna().median() - 1), 3)
                  for a, b in [(2022, 2023), (2023, 2024)] if a in by and b in by}
    stats["yoy_change_jan_aug_median_store"] = yoy


def weekly(df: pd.DataFrame) -> None:
    d = df[df.trading & ~df.holiday].groupby(["client", "dow"]).idx.mean().unstack(0)
    d = d / d.mean()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for c in CLIENTS:
        ax.plot(DOW, d[c].values, color=CLIENT_COLORS[c], marker="o", markersize=6, label=CLIENT_LABELS[c])
        label_end(ax, 6, d[c].iloc[-1], c)
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.set_title("Weekly pattern (index: average day = 1)")
    ax.set_ylabel("Sales index")
    ax.legend(loc="upper left", ncols=3)
    save(fig, "02_weekly")
    stats["weekly_index"] = d.round(3).to_dict()


def yearly(df: pd.DataFrame) -> None:
    d = df[df.trading & ~df.holiday].groupby(["client", "month"]).idx.mean().unstack(0)
    d = d / d.mean()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for c in CLIENTS:
        ax.plot(months, d[c].values, color=CLIENT_COLORS[c], marker="o", markersize=6, label=CLIENT_LABELS[c])
        label_end(ax, 11, d[c].iloc[-1], c)
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.set_title("Yearly pattern, holidays excluded (index: average month = 1)")
    ax.set_ylabel("Sales index")
    ax.legend(loc="upper left", ncols=3)
    save(fig, "03_yearly")
    stats["monthly_index"] = d.round(3).to_dict()


def holidays(df: pd.DataFrame) -> None:
    rows = []
    for kind in ("public_holiday", "retail_event"):
        d = df[df[kind].notna() & df.lift.notna()].copy()
        # Substitute days (a holiday falling on a weekend) behave differently from the day itself.
        first = d[kind].str.split(r" \| |; ").str[0]
        d["name"] = first.where(~d[kind].str.contains("observed"), "Substitute holiday (observed)")
        g = d.groupby(["client", "name"]).agg(
            lift=("lift", "mean"), closed=("is_closed", "mean"), n=("lift", "size"))
        rows.append(g.assign(kind=kind))
    h = pd.concat(rows).reset_index()
    h = h[h.n >= 20]
    stats["holiday_effects"] = {
        c: h[h.client == c].set_index("name")[["lift", "closed", "n"]].round(3).to_dict("index") for c in CLIENTS
    }

    order = h.groupby("name").lift.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(order) + 1.2))
    y = {n: i for i, n in enumerate(order)}
    for i, c in enumerate(CLIENTS):
        d = h[h.client == c]
        ax.scatter(d.lift, [y[n] + (i - 1) * 0.22 for n in d.name], s=36, color=CLIENT_COLORS[c],
                   label=CLIENT_LABELS[c], edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(range(len(order)), order)
    ax.axvline(1, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Sales vs a normal same weekday (1 = no effect; closures count as 0)")
    ax.set_title("Effect of holidays and retail events")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "04_holidays")


def christmas(df: pd.DataFrame) -> None:
    d = df[df.lift.notna()].copy()
    md = d.date.dt.strftime("%m-%d")
    days = [f"12-{x:02d}" for x in range(10, 32)] + [f"01-{x:02d}" for x in range(1, 8)]
    d = d[md.isin(days)].assign(md=md)
    g = d.groupby(["client", "md"]).lift.mean().unstack(0).reindex(days)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for c in CLIENTS:
        ax.plot(range(len(days)), g[c].values, color=CLIENT_COLORS[c], marker="o", markersize=4,
                label=CLIENT_LABELS[c])
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ticks = [days.index(t) for t in ["12-10", "12-17", "12-24", "12-31", "01-07"]]
    ax.set_xticks(ticks, ["10 Dec", "17 Dec", "24 Dec", "31 Dec", "7 Jan"])
    ax.set_title("Christmas and New Year (sales vs a normal same weekday)")
    ax.set_ylabel("Lift")
    ax.legend(loc="upper left", ncols=3)
    save(fig, "05_christmas")
    stats["christmas_lift"] = g.round(3).to_dict()


def weather(df: pd.DataFrame) -> None:
    d = df[df.trading & ~df.holiday & df.lift.notna()].copy()
    store = [d.client, d.store_id]
    # Temperature relative to the local norm for the time of year (centred 31-day mean).
    norm = d.groupby(store).temp_max_c.transform(lambda s: s.rolling(31, center=True, min_periods=15).mean())
    d["temp_anom"] = d.temp_max_c - norm
    d["log_lift"] = np.log(d.lift)

    # The original project's approach: raw sales against raw weather.
    naive = {c: d[d.client == c][["sales", "temp_max_c", "precip_mm", "wind_speed_kmh"]].corr().sales
             .drop("sales").round(3).to_dict() for c in CLIENTS}
    # Store-level means first, so the pooled correlation is not driven by store size.
    deseason = {c: d[d.client == c][["log_lift", "temp_anom", "precip_mm", "wind_speed_kmh"]].corr().log_lift
                .drop("log_lift").round(3).to_dict() for c in CLIENTS}
    stats["weather_corr_naive_sales"] = naive
    stats["weather_corr_deseasonalised_log_lift"] = deseason

    precip_bins = [-0.1, 0.1, 1, 5, 10, 20, 200]
    precip_labels = ["dry", "0.1-1", "1-5", "5-10", "10-20", "20+"]
    d["precip_bin"] = pd.cut(d.precip_mm, precip_bins, labels=precip_labels)
    temp_bins = [-30, -6, -3, -1, 1, 3, 6, 30]
    temp_labels = ["< -6", "-6 to -3", "-3 to -1", "-1 to 1", "1 to 3", "3 to 6", "> 6"]
    d["temp_bin"] = pd.cut(d.temp_anom, temp_bins, labels=temp_labels)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    out = {}
    for ax, col, labels, title, xlabel in [
        (axes[0], "precip_bin", precip_labels, "Rain", "Daily precipitation (mm)"),
        (axes[1], "temp_bin", temp_labels, "Temperature vs normal", "Max temperature minus local norm (°C)"),
    ]:
        g = d.groupby(["client", col], observed=True).lift.agg(["mean", "size"]).reset_index()
        g = g[g["size"] >= 200]
        for c in CLIENTS:
            s = g[g.client == c].set_index(col)["mean"].reindex(labels)
            ax.plot(labels, s.values, color=CLIENT_COLORS[c], marker="o", markersize=6, label=CLIENT_LABELS[c])
        ax.axhline(1, color=MUTED, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        out[col] = g.pivot(index=col, columns="client", values="mean").round(3).to_dict()
    axes[0].set_ylabel("Sales vs a normal same weekday")
    axes[0].legend(loc="lower left")
    save(fig, "06_weather")
    stats["weather_lift_by_bin"] = out


def autocorrelation(df: pd.DataFrame) -> None:
    lags = list(range(1, 29))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, layout="constrained")
    out = {}
    rng = np.random.default_rng(0)
    for ax, c in zip(axes, CLIENTS):
        d = df[df.client == c]
        ids = d.store_id.unique()
        ids = rng.choice(ids, size=min(150, len(ids)), replace=False)
        acf = []
        for sid in ids:
            s = np.log(d[d.store_id == sid].idx.where(lambda x: x > 0)).reset_index(drop=True)
            acf.append([s.autocorr(k) for k in lags + [364]])
        m = np.nanmean(acf, axis=0)
        ax.bar(lags, m[:-1], color=CLIENT_COLORS[c], width=0.7)
        ax.set_title(CLIENT_LABELS[c])
        ax.set_xticks([1, 7, 14, 21, 28])
        ax.set_xlabel("Lag (days)")
        out[c] = {"lag1": round(float(m[0]), 3), "lag7": round(float(m[6]), 3),
                  "lag14": round(float(m[13]), 3), "lag28": round(float(m[27]), 3),
                  "lag364": round(float(m[-1]), 3)}
    axes[0].set_ylabel("Autocorrelation of log sales")
    fig.suptitle("How much today's sales echo earlier days", x=0.01, ha="left", fontweight="bold", fontsize=12)
    save(fig, "07_autocorrelation")
    stats["autocorrelation"] = out


def store_differences(df: pd.DataFrame) -> None:
    d = df[df.trading & ~df.holiday]
    wk = d.assign(weekend=d.dow >= 5).groupby(["client", "store_id", "weekend"]).idx.mean().unstack()
    ratio = (wk[True] / wk[False]).rename("weekend_ratio").reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True, layout="constrained")
    out = {}
    for ax, c in zip(axes, CLIENTS):
        r = ratio[ratio.client == c].weekend_ratio.dropna()
        ax.hist(r, bins=np.arange(0, 2.6, 0.1), color=CLIENT_COLORS[c], edgecolor="white", linewidth=1)
        ax.axvline(1, color=MUTED, linewidth=0.8)
        ax.set_title(f"{CLIENT_LABELS[c]} ({len(r)} stores)")
        ax.set_xlabel("Weekend day vs weekday sales")
        out[c] = {"median": round(float(r.median()), 3), "p10": round(float(r.quantile(0.1)), 3),
                  "p90": round(float(r.quantile(0.9)), 3), "share_below_0.8": round(float((r < 0.8).mean()), 3),
                  "share_above_1.2": round(float((r > 1.2).mean()), 3)}
    axes[0].set_ylabel("Stores")
    fig.suptitle("Stores differ: weekend vs weekday trade", x=0.01, ha="left", fontweight="bold", fontsize=12)
    save(fig, "08_store_weekend_ratio")
    stats["store_weekend_ratio"] = out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    stats["rows"] = int(len(df))
    stats["lift_coverage"] = round(float(df.lift.notna().mean()), 3)
    for step in (trend, weekly, yearly, holidays, christmas, weather, autocorrelation, store_differences):
        step(df)
        print("done:", step.__name__, flush=True)
    (REPORTS / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
