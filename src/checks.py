"""Data checks on the processed tables. Each check guards against an issue found in the raw exports.

Run: python -m src.checks   (also runs at the end of src.clean)
"""
import json

import pandas as pd

from src.config import PROCESSED, START_DATE


def run(sales: pd.DataFrame, stores: pd.DataFrame, log: dict) -> list[str]:
    failures = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            failures.append(message)

    check(not sales.duplicated(["client", "store_id", "date"]).any(), "duplicate client/store/date rows")
    check((sales.sales.dropna() >= 0).all(), "negative sales")
    check(sales.date.min() >= START_DATE, f"rows before {START_DATE.date()} (COVID period)")

    # Zero placeholder rows after each client's extract date must be gone.
    for client, entry in log.items():
        end = pd.Timestamp(entry["data_end"])
        check(sales.loc[sales.client == client, "date"].max() <= end, f"{client}: rows after data end {end.date()}")

    keys = ["client", "store_id"]
    check(
        sales[keys].drop_duplicates().merge(stores[keys], how="outer", indicator=True)._merge.eq("both").all(),
        "stores table and sales table list different stores",
    )

    check(stores.region.isin(["ON", "ENG", "WLS", "SCT"]).all(), "store without a holiday region")

    # Each store's trading period starts and ends with a sale.
    first_last = sales.groupby(keys).agg(first=("date", "min"), last=("date", "max")).reset_index()
    m = first_last.merge(stores, on=keys)
    check(((m["first"] == m.first_sale) & (m["last"] == m.last_sale)).all(), "trading period mismatch")

    # Regression check for the original merge bug: client-level special days must reach every store.
    for client, day in [("costa", "2023-12-25"), ("moto", "2022-12-25")]:
        on_day = sales[(sales.client == client) & (sales.date == day)]
        check(
            on_day.special_day.fillna("").str.contains("Christmas").all(),
            f"{client}: Christmas {day} missing for some stores",
        )
    return failures


def main() -> None:
    sales = pd.read_parquet(PROCESSED / "sales_daily.parquet")
    stores = pd.read_parquet(PROCESSED / "stores.parquet")
    log = json.loads((PROCESSED / "quality_log.json").read_text())
    failures = run(sales, stores, log)
    for f in failures:
        print("FAIL:", f)
    if failures:
        raise SystemExit(1)
    print("all data checks passed")


if __name__ == "__main__":
    main()
