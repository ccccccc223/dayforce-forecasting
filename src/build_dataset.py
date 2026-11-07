"""Join cleaned sales with store info, holidays and weather into one analysis table.

Output: data/processed/analysis_daily.parquet, one row per client x store x day, with
  sales, is_closed, is_anomaly      from sales_daily
  client_special_day                the client's own special-day entry (may be missing)
  public_holiday, retail_event      official calendar for the store's region
  temp_max_c ... wind_dir_deg       daily weather (Longo's by postal code, UK by grid cell)
  dow, month, year                  calendar fields
Run: python -m src.build_dataset
"""
import pandas as pd

from src.config import PROCESSED

WEATHER_COLS = [
    "temp_max_c", "temp_min_c", "precip_mm", "rain_mm", "snow_cm",
    "wind_speed_kmh", "wind_gust_kmh", "wind_dir_deg",
]


def holiday_columns() -> pd.DataFrame:
    """(region, date) -> public_holiday / retail_event names; several on a day are joined."""
    cal = pd.read_parquet(PROCESSED / "holidays.parquet")
    out = cal.pivot_table(index=["region", "date"], columns="kind", values="name",
                          aggfunc=lambda s: " | ".join(sorted(s)))
    out = out.rename(columns={"public": "public_holiday", "event": "retail_event"})
    return out.reset_index()[["region", "date", "public_holiday", "retail_event"]]


def weather_by_store() -> pd.DataFrame:
    """(client, store_id, date) -> weather, for every store."""
    stores = pd.read_parquet(PROCESSED / "stores.parquet")

    longos = stores[stores.client == "longos"][["client", "store_id", "postal_code"]]
    w_longos = pd.read_parquet(PROCESSED / "weather_longos.parquet")
    longos = longos.merge(w_longos, on="postal_code").drop(columns="postal_code")

    cells = pd.read_parquet(PROCESSED / "store_weather_cell.parquet")[["client", "store_id", "cell"]]
    w_uk = pd.read_parquet(PROCESSED / "weather_uk.parquet")
    uk = cells.merge(w_uk, on="cell").drop(columns="cell")

    return pd.concat([longos, uk], ignore_index=True)[["client", "store_id", "date", *WEATHER_COLS]]


def main() -> None:
    sales = pd.read_parquet(PROCESSED / "sales_daily.parquet")
    stores = pd.read_parquet(PROCESSED / "stores.parquet")[["client", "store_id", "region"]]

    df = sales.rename(columns={"special_day": "client_special_day"})
    df = df.merge(stores, on=["client", "store_id"], how="left")
    df = df.merge(holiday_columns(), on=["region", "date"], how="left")
    df = df.merge(weather_by_store(), on=["client", "store_id", "date"], how="left")

    df["dow"] = df.date.dt.dayofweek  # Monday = 0
    df["month"] = df.date.dt.month
    df["year"] = df.date.dt.year
    df = df[[
        "client", "store_id", "region", "date", "dow", "month", "year",
        "sales", "is_closed", "is_anomaly",
        "client_special_day", "public_holiday", "retail_event", *WEATHER_COLS,
    ]]

    assert len(df) == len(sales), "join changed the row count"
    assert not df.duplicated(["client", "store_id", "date"]).any(), "duplicate rows after join"
    assert df.region.notna().all(), "store without region"
    missing = df[WEATHER_COLS].isna().any(axis=1)
    assert not missing.any(), f"{missing.sum()} rows without weather"

    df.to_parquet(PROCESSED / "analysis_daily.parquet", index=False)
    print(f"analysis_daily: {len(df):,} rows x {df.shape[1]} cols")
    summary = df.groupby("client").agg(
        rows=("sales", "size"),
        public_holiday_rows=("public_holiday", "count"),
        retail_event_rows=("retail_event", "count"),
        client_special_rows=("client_special_day", "count"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
