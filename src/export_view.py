"""Export the processed tables to files that open in Excel / Numbers, for inspection.

data/view/processed_tables.xlsx  one sheet per table; Costa limited to its first
                                 SAMPLE_STORES stores to keep the workbook light
data/view/analysis_<client>.csv  full merged daily table (sales + holidays + weather) per client
Run: python -m src.export_view
"""
import pandas as pd

from src.config import PROCESSED, ROOT

VIEW = ROOT / "data" / "view"
SAMPLE_STORES = 20


def main() -> None:
    VIEW.mkdir(parents=True, exist_ok=True)
    for old in VIEW.glob("sales_*.csv"):  # superseded by analysis_*.csv
        old.unlink()
    daily = pd.read_parquet(PROCESSED / "analysis_daily.parquet")
    for client, df in daily.groupby("client"):
        df.to_csv(VIEW / f"analysis_{client}.csv", index=False, date_format="%Y-%m-%d")

    costa = daily[daily.client == "costa"]
    costa_sample = costa[costa.store_id.isin(sorted(costa.store_id.unique())[:SAMPLE_STORES])]
    sheets = {
        "stores": pd.read_parquet(PROCESSED / "stores.parquet"),
        "holidays": pd.read_parquet(PROCESSED / "holidays.parquet"),
        "analysis_longos": daily[daily.client == "longos"],
        "analysis_moto": daily[daily.client == "moto"],
        f"analysis_costa_{SAMPLE_STORES}_stores": costa_sample,
    }
    with pd.ExcelWriter(VIEW / "processed_tables.xlsx", datetime_format="yyyy-mm-dd") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name, index=False)
            xl.sheets[name].freeze_panes = "A2"
            print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
