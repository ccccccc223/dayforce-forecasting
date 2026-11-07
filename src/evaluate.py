"""Score the backtest forecasts.

WAPE = sum |actual - forecast| / sum actual, on trading days only (closures are planned and
known, so they are not forecast). Bias = sum (forecast - actual) / sum actual.
Two comparisons:
  all stores    baselines and LightGBM variants on every store
  stat sample   adds Prophet and DHR, on the 204-store sample they were run on
Output: reports/metrics.json, reports/figures/09_*.png, 10_*.png
Run: python -m src.evaluate
"""
import json

import numpy as np
import pandas as pd

from src.backtest import OUT
from src.config import PROCESSED, REPORTS
from src.plotstyle import CLIENT_COLORS, CLIENT_LABELS, CLIENTS, INK_2, MUTED, plt

MODEL_LABELS = {
    "naive": "Seasonal naive", "ma": "Moving average (Dayforce MA)", "toha": "ToHA (Dayforce)",
    "prophet": "Prophet", "dhr": "Dynamic harmonic regression",
    "lgbm": "LightGBM", "lgbm_noweather": "LightGBM, no weather", "lgbm_noholiday": "LightGBM, no holiday features",
}


def load(client: str) -> pd.DataFrame:
    d = pd.read_parquet(OUT / f"{client}.parquet")
    stats_path = OUT / f"stats_{client}.parquet"
    if stats_path.exists():
        d = d.merge(pd.read_parquet(stats_path).drop(columns="origin"), on=["store_id", "fold", "date"], how="left")
    days = pd.read_parquet(PROCESSED / "analysis_daily.parquet",
                           columns=["client", "store_id", "date", "public_holiday", "retail_event"])
    days = days[days.client == client].drop(columns="client")
    d = d.merge(days, on=["store_id", "date"], how="left")
    d["special"] = d.public_holiday.notna() | d.retail_event.notna()
    return d[d.trading].assign(client=client, week=lambda x: (x.h - 1) // 7 + 1)


def wape(d: pd.DataFrame, model: str) -> float:
    return float((d.actual - d[f"pred_{model}"]).abs().sum() / d.actual.sum())


def bias(d: pd.DataFrame, model: str) -> float:
    return float((d[f"pred_{model}"] - d.actual).sum() / d.actual.sum())


def score(d: pd.DataFrame, models: list[str]) -> dict:
    d = d.dropna(subset=[f"pred_{m}" for m in models])
    return {
        "rows": int(len(d)), "stores": int(d.store_id.nunique()),
        "wape": {m: round(wape(d, m), 4) for m in models},
        "bias": {m: round(bias(d, m), 4) for m in models},
        "wape_by_week": {m: {int(w): round(wape(g, m), 4) for w, g in d.groupby("week")} for m in models},
        "wape_special_days": {m: round(wape(d[d.special], m), 4) for m in models},
        "wape_normal_days": {m: round(wape(d[~d.special], m), 4) for m in models},
    }


def plot_models(metrics: dict, key: str, models: list[str], name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(models) + 1.3))
    y = np.arange(len(models))
    for i, c in enumerate(CLIENTS):
        if c not in metrics[key]:
            continue
        w = [metrics[key][c]["wape"][m] * 100 for m in models]
        ax.scatter(w, y + (i - 1) * 0.2, s=40, color=CLIENT_COLORS[c], label=CLIENT_LABELS[c],
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y, [MODEL_LABELS[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("WAPE (%), lower is better")
    ax.set_xlim(left=0)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.savefig(REPORTS / "figures" / f"{name}.png")
    plt.close(fig)


def plot_horizon(metrics: dict, models: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, layout="constrained")
    greys = {"naive": "#c3c2b7", "ma": "#a3a29b", "toha": MUTED}
    for ax, c in zip(axes, CLIENTS):
        byw = metrics["all_stores"][c]["wape_by_week"]
        ends = []
        for m in models:
            vals = [byw[m][w] * 100 for w in (1, 2, 3, 4)]
            color = CLIENT_COLORS[c] if m == "lgbm" else greys.get(m, MUTED)
            ax.plot([1, 2, 3, 4], vals, color=color, marker="o", markersize=5, linewidth=2 if m == "lgbm" else 1.4)
            ends.append([vals[-1], MODEL_LABELS[m].split(" (")[0]])
        # End labels, nudged apart so close lines stay readable.
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.55)
        for (y, label), m in zip(ends, sorted(models, key=lambda m: byw[m][4])):
            ax.annotate(label, (4, byw[m][4] * 100), xytext=(4.12, y), textcoords="data",
                        va="center", fontsize=8, color=INK_2)
        ax.set_title(CLIENT_LABELS[c])
        ax.set_xticks([1, 2, 3, 4], ["wk 1", "wk 2", "wk 3", "wk 4"])
        ax.set_xlim(0.8, 5.2)
    axes[0].set_ylabel("WAPE (%)")
    fig.suptitle("Error by forecast week (all stores)", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.savefig(REPORTS / "figures" / "10_wape_by_horizon.png")
    plt.close(fig)


def main() -> None:
    all_models = ["naive", "ma", "toha", "lgbm", "lgbm_noweather", "lgbm_noholiday"]
    stat_models = ["naive", "ma", "toha", "prophet", "dhr", "lgbm"]
    metrics = {"all_stores": {}, "stat_sample": {}}
    for c in CLIENTS:
        d = load(c)
        metrics["all_stores"][c] = score(d, all_models)
        if "pred_dhr" in d:
            sample = d[d.pred_dhr.notna()]
            metrics["stat_sample"][c] = score(sample, stat_models)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_models(metrics, "all_stores", all_models, "09_wape_all_stores", "Forecast error, all stores (28-day horizon)")
    if metrics["stat_sample"]:
        plot_models(metrics, "stat_sample", stat_models, "09b_wape_stat_sample",
                    "Forecast error incl. statistical models (204-store sample)")
    plot_horizon(metrics, ["naive", "ma", "toha", "lgbm"])

    for key, block in metrics.items():
        for c, m in block.items():
            print(key, c, m["stores"], "stores |", " ".join(f"{k}={v:.3f}" for k, v in m["wape"].items()))


if __name__ == "__main__":
    main()
