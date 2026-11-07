"""Cache the raw Excel exports as parquet, unchanged apart from column names as strings.

Reading the 53 MB Costa workbook takes minutes; parquet loads in about a second.
Run: python -m src.ingest
"""
import pandas as pd

from src.config import INTERIM, RAW, RAW_FILES


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    for name, (filename, sheet) in RAW_FILES.items():
        df = pd.read_excel(RAW / filename, sheet_name=sheet)
        df.columns = [str(c) for c in df.columns]
        # Text columns can hold stray Excel-converted dates; keep them as text.
        for col in df.select_dtypes("object"):
            df[col] = df[col].astype("string")
        df.to_parquet(INTERIM / f"{name}.parquet", index=False)
        print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} cols")


if __name__ == "__main__":
    main()
