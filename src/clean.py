"""Clean the three clients' daily sales into one schema.

Outputs (data/processed):
  sales_daily.parquet   one row per client x store x day inside the store's trading period
  stores.parquet        one row per store, with trading period and status
  weather_longos.parquet  daily weather by postal code (Longo's only; UK weather is fetched later)
  quality_log.json      counts behind reports/data_quality.md

Run: python -m src.clean
"""
import json
import re

import numpy as np
import pandas as pd

from src import checks
from src.config import INTERIM, PROCESSED, START_DATE

# Longo's org units that are not retail stores (distribution centre, kitchens,
# training store, online grocery). Excluded from store-level forecasting.
LONGOS_NON_STORES = {
    "950-DC", "801-Central Kitchen", "099-Training Store", "900-Vaughan",
    "750-Grocery Gateway", "752 Voila Commissary", "850-Market Cafe", "501-Longo's Pronto Eats",
}
# Special days assigned to these parent org units apply to every store of the client.
CLIENT_ROOT_ORGS = {"costa": 132436, "moto": 1000}

# Relative to the store's centred 29-day median: above ANOMALY_FACTOR is a data spike,
# below CLOSED_FACTOR the store was effectively closed.
ANOMALY_FACTOR = 4.0
CLOSED_FACTOR = 0.05

# UK postcode areas outside England (bank holidays differ in Scotland).
SCOTLAND_AREAS = {"AB", "DD", "DG", "EH", "FK", "G", "HS", "IV", "KA", "KW", "KY", "ML", "PA", "PH", "TD", "ZE"}
WALES_AREAS = {"CF", "LL", "NP", "SA", "LD"}

LONGOS_WEATHER = {
    "Max Temperature (°C)": "temp_max_c",
    "Min Temperature (°C)": "temp_min_c",
    "Precipitation (mm)": "precip_mm",
    "Rain (mm)": "rain_mm",
    "Snowfall (cm)": "snow_cm",
    "Wind Speed (km/h)": "wind_speed_kmh",
    "Wind Gust (km/h)": "wind_gust_kmh",
    "Wind Direction (°)": "wind_dir_deg",
}

log: dict = {}


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(INTERIM / f"{name}.parquet")


def region(client: str, postal_code: str) -> str:
    """Holiday region of a store: Ontario, or England / Wales / Scotland from the postcode area."""
    if client == "longos":
        return "ON"
    area = re.match(r"[A-Z]+", postal_code.upper()).group()
    return "SCT" if area in SCOTLAND_AREAS else "WLS" if area in WALES_AREAS else "ENG"


def data_end(sales: pd.DataFrame) -> pd.Timestamp:
    """Last day on which a normal share of stores reported positive sales.

    After the extract date the source system holds zero placeholder rows, so the
    share of selling stores collapses; partial last days are excluded too.
    """
    share = sales.assign(pos=sales.sales > 0).groupby("date").pos.mean()
    return share[share >= 0.9 * share.median()].index.max()


def special_days_by_store(special: pd.DataFrame, client: str, store_ids) -> pd.DataFrame:
    """Expand client special-day assignments to (store_id, date, special_day)."""
    special = special.rename(columns={"DayDate": "date", "ShortName": "special_day"})
    special = special[special.date >= START_DATE]
    root = CLIENT_ROOT_ORGS[client]
    stores = pd.DataFrame({"store_id": sorted(store_ids)})

    at_store = special[special.OrgUnitId.isin(store_ids)].rename(columns={"OrgUnitId": "store_id"})
    at_root = special[special.OrgUnitId == root].drop(columns="OrgUnitId").merge(stores, how="cross")
    unmapped = special[~special.OrgUnitId.isin(store_ids) & (special.OrgUnitId != root)]
    log[client]["special_rows_store_level"] = len(at_store)
    log[client]["special_rows_client_level"] = int((special.OrgUnitId == root).sum())
    log[client]["special_rows_unmapped_org"] = len(unmapped)
    log[client]["unmapped_orgs"] = sorted(int(o) for o in unmapped.OrgUnitId.unique())

    out = pd.concat([at_store, at_root])[["store_id", "date", "special_day"]].drop_duplicates()
    # Several special days can fall on one store-day; keep them all in one field.
    return out.groupby(["store_id", "date"], as_index=False).special_day.agg(
        lambda s: " | ".join(sorted(s))
    )


def load_costa() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load("costa_sales")
    sales = raw.rename(columns={
        "OrgUnitId": "store_id", "DayId": "date", "Value": "sales",
        "LongName_x": "store_name", "PostalCode": "postal_code",
    })[["store_id", "date", "sales", "store_name", "postal_code"]]
    special = special_days_by_store(load("costa_specialdays"), "costa", set(sales.store_id))
    return sales, special


def load_moto() -> tuple[pd.DataFrame, pd.DataFrame]:
    # The "merged" Moto file is exactly this history filtered to 2022+, so use the history.
    raw = load("moto_sales_history")
    sales = raw.rename(columns={
        "OrgId": "store_id", "DayId": "date", "Value": "sales",
        "LongName": "store_name", "PostalCode": "postal_code",
    })[["store_id", "date", "sales", "store_name", "postal_code"]]
    special = special_days_by_store(load("moto_specialdays"), "moto", set(sales.store_id))
    return sales, special


def load_longos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load("longos_sales_weather")
    # The only duplicated store-days are on the 2025-01-07 placeholder date.
    raw = raw.drop_duplicates(["OrgUnitId", "DayId"])
    non_store = raw.Address_ShortName.isin(LONGOS_NON_STORES)
    log["longos"]["non_store_units_excluded"] = sorted(raw[non_store].Address_ShortName.unique())
    raw = raw[~non_store]
    sales = raw.rename(columns={
        "OrgUnitId": "store_id", "DayId": "date", "Total_Sales_Value": "sales",
        "Address_ShortName": "store_name", "PostalCode": "postal_code",
    })
    special = sales[sales.SpecialDayShortName.notna()][["store_id", "date", "SpecialDayShortName"]]
    special = special.rename(columns={"SpecialDayShortName": "special_day"})
    weather = raw.rename(columns={"PostalCode": "postal_code", "DayId": "date", **LONGOS_WEATHER})
    weather = weather[["postal_code", "date", *LONGOS_WEATHER.values()]].drop_duplicates(["postal_code", "date"])
    return sales[["store_id", "date", "sales", "store_name", "postal_code"]], special, weather


def clean_client(client: str, sales: pd.DataFrame, special: pd.DataFrame):
    entry = log[client]
    entry["raw_rows"] = len(sales)
    entry["raw_stores"] = int(sales.store_id.nunique())
    sales = sales[sales.date >= START_DATE]

    end = data_end(sales)
    entry["data_end"] = str(end.date())
    entry["placeholder_rows_after_end"] = int((sales.date > end).sum())
    sales = sales[sales.date <= end]

    entry["negative_sales_rows"] = int((sales.sales < 0).sum())
    sales = sales.assign(sales=sales.sales.clip(lower=0))

    # Trading period: first to last day with positive sales.
    pos = sales[sales.sales > 0]
    period = pos.groupby("store_id").date.agg(first_sale="min", last_sale="max")
    entry["stores_never_trading"] = int(sales.store_id.nunique() - len(period))

    # Re-index each store to a full daily calendar so missing days become explicit.
    frames = []
    for sid, (first, last) in period.iterrows():
        days = pd.DataFrame({"store_id": sid, "date": pd.date_range(first, last, freq="D")})
        frames.append(days)
    grid = pd.concat(frames)
    df = grid.merge(sales[["store_id", "date", "sales"]], on=["store_id", "date"], how="left")
    entry["missing_days_in_period"] = int(df.sales.isna().sum())

    df = df.merge(special, on=["store_id", "date"], how="left")

    # Sales relative to the store's centred 29-day median of trading days.
    trading = df.sales.where(df.sales > 0)
    med = trading.groupby(df.store_id).transform(
        lambda s: s.rolling(29, center=True, min_periods=7).median()
    )
    ratio = df.sales / med
    # A few pounds/dollars of sales (stray transactions) means the store was effectively shut.
    df["is_closed"] = ((df.sales == 0) | (ratio < CLOSED_FACTOR)).astype(int)
    entry["near_zero_rows_marked_closed"] = int(((df.sales > 0) & (ratio < CLOSED_FACTOR)).sum())
    # Only spikes are anomalies: low days are mostly holidays, which the model should learn.
    df["is_anomaly"] = (ratio > ANOMALY_FACTOR).astype(int)
    entry["anomaly_rows"] = int(df.is_anomaly.sum())
    entry["closed_rows"] = int(df.is_closed.sum())
    entry["special_day_rows"] = int(df.special_day.notna().sum())
    entry["clean_rows"] = len(df)

    info = sales.drop_duplicates("store_id").set_index("store_id")[["store_name", "postal_code"]]
    stores = period.join(info).reset_index()
    stores["client"] = client
    stores["region"] = [region(client, pc) for pc in stores.postal_code]
    # A store whose sales stop well before the client's data end has closed (or, for
    # the last Costa store in its workbook, was cut off by Excel's row limit).
    stores["ended_early"] = (stores.last_sale < end - pd.Timedelta(days=28)).astype(int)
    stores["days"] = (stores.last_sale - stores.first_sale).dt.days + 1
    entry["stores_ended_early"] = int(stores.ended_early.sum())
    entry["clean_stores"] = len(stores)

    df.insert(0, "client", client)
    return df, stores


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for c in ("longos", "moto", "costa"):
        log[c] = {}

    longos_sales, longos_special, weather = load_longos()
    inputs = {
        "longos": (longos_sales, longos_special),
        "moto": load_moto(),
        "costa": load_costa(),
    }
    days, stores = [], []
    for client, (sales, special) in inputs.items():
        d, s = clean_client(client, sales, special)
        days.append(d)
        stores.append(s)

    sales_daily = pd.concat(days, ignore_index=True)
    sales_daily["store_id"] = sales_daily.store_id.astype(np.int64)
    stores = pd.concat(stores, ignore_index=True)[
        ["client", "store_id", "store_name", "postal_code", "region", "first_sale", "last_sale", "days", "ended_early"]
    ]

    weather = weather[weather.date >= START_DATE]
    log["longos"]["weather_missing_cells"] = int(weather.drop(columns=["postal_code", "date"]).isna().sum().sum())

    sales_daily.to_parquet(PROCESSED / "sales_daily.parquet", index=False)
    stores.to_parquet(PROCESSED / "stores.parquet", index=False)
    weather.to_parquet(PROCESSED / "weather_longos.parquet", index=False)
    (PROCESSED / "quality_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log, indent=2, default=str))

    failures = checks.run(sales_daily, stores, json.loads(json.dumps(log, default=str)))
    if failures:
        raise SystemExit("data checks failed:\n" + "\n".join(failures))
    print("all data checks passed")


if __name__ == "__main__":
    main()
