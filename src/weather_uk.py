"""Fetch daily historical weather for UK stores from the Open-Meteo archive API.

Stores are grouped into grid cells (GRID_DEG) so nearby stores share one request:
the free API allows 10,000 weighted calls per day, and a ~3-year daily request costs
about 70 calls per location. Each cell is cached on disk, so the script can be
stopped and rerun; it pauses when the API reports a rate limit.

Output: data/interim/weather_uk_cells/<cell>.parquet, data/processed/weather_uk.parquet,
        data/processed/store_weather_cell.parquet (store -> cell)
Run: python -m src.weather_uk [max_cells]
"""
import sys
import time

import numpy as np
import pandas as pd
import requests

from src.config import INTERIM, PROCESSED, START_DATE

API = "https://archive-api.open-meteo.com/v1/archive"
GRID_DEG = 0.5
END_DATE = pd.Timestamp("2024-08-25")  # latest UK data end (Costa)

# Same variables, units and names as the Longo's weather data.
DAILY = {
    "temperature_2m_max": "temp_max_c",
    "temperature_2m_min": "temp_min_c",
    "precipitation_sum": "precip_mm",
    "rain_sum": "rain_mm",
    "snowfall_sum": "snow_cm",
    "wind_speed_10m_max": "wind_speed_kmh",
    "wind_gusts_10m_max": "wind_gust_kmh",
    "wind_direction_10m_dominant": "wind_dir_deg",
}
CACHE = INTERIM / "weather_uk_cells"


def store_cells() -> pd.DataFrame:
    stores = pd.read_parquet(PROCESSED / "stores.parquet")
    uk = stores[stores.client.isin(["moto", "costa"])].copy()
    uk["postal_code"] = uk.postal_code.str.upper().str.replace(" ", "")
    geo = pd.read_parquet(INTERIM / "uk_postcodes.parquet")
    uk = uk.merge(geo, on="postal_code", how="left")
    snap_lat = (np.round(uk.lat / GRID_DEG) * GRID_DEG).round(2)
    snap_lon = (np.round(uk.lon / GRID_DEG) * GRID_DEG).round(2)
    uk["cell"] = snap_lat.astype(str) + "_" + snap_lon.astype(str)
    # Query at the mean position of the cell's stores, not the cell centre, which can
    # lie well away from them (or over the sea).
    uk["cell_lat"] = uk.groupby("cell").lat.transform("mean").round(4)
    uk["cell_lon"] = uk.groupby("cell").lon.transform("mean").round(4)
    return uk[["client", "store_id", "postal_code", "lat", "lon", "cell", "cell_lat", "cell_lon"]]


def fetch_cell(lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(START_DATE.date()), "end_date": str(END_DATE.date()),
        "daily": ",".join(DAILY), "timezone": "Europe/London",
    }
    while True:
        r = requests.get(API, params=params, timeout=120)
        if r.status_code == 429:
            reason = r.json().get("reason", "")
            wait = 3600 if "Hourly" in reason else 60 if "Minutely" in reason else None
            if wait is None:
                raise SystemExit(f"stopping, rerun later: {reason}")
            print(f"rate limited ({reason}); waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        daily = r.json()["daily"]
        df = pd.DataFrame(daily).rename(columns={"time": "date", **DAILY})
        df["date"] = pd.to_datetime(df.date)
        return df


def main() -> None:
    max_cells = int(sys.argv[1]) if len(sys.argv) > 1 else None
    CACHE.mkdir(parents=True, exist_ok=True)
    stores = store_cells()
    cells = stores.drop_duplicates("cell")[["cell", "cell_lat", "cell_lon"]].sort_values("cell")
    todo = [c for c in cells.itertuples() if not (CACHE / f"{c.cell}.parquet").exists()]
    print(f"{len(cells)} cells, {len(todo)} still to fetch", flush=True)

    for i, c in enumerate(todo[:max_cells], 1):
        fetch_cell(c.cell_lat, c.cell_lon).to_parquet(CACHE / f"{c.cell}.parquet", index=False)
        print(f"[{i}/{len(todo)}] {c.cell}", flush=True)

    done = [p for p in CACHE.glob("*.parquet")]
    if len(done) < len(cells):
        print(f"{len(done)}/{len(cells)} cells cached; rerun to continue")
        return
    weather = pd.concat(pd.read_parquet(p).assign(cell=p.stem) for p in done)
    weather.to_parquet(PROCESSED / "weather_uk.parquet", index=False)
    stores.to_parquet(PROCESSED / "store_weather_cell.parquet", index=False)
    print(f"weather_uk: {len(weather):,} rows for {weather.cell.nunique()} cells")


if __name__ == "__main__":
    main()
