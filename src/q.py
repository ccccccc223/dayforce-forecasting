"""Query the parquet files with SQL from the terminal.

Tables: sales (sales_daily), stores, weather (weather_longos).
Example: .venv/bin/python -m src.q "SELECT client, COUNT(*) FROM sales GROUP BY 1"
"""
import sys

import duckdb

from src.config import PROCESSED

con = duckdb.connect()
for view, file in [("sales", "sales_daily"), ("stores", "stores"), ("weather", "weather_longos")]:
    con.execute(f"CREATE VIEW {view} AS SELECT * FROM '{PROCESSED / file}.parquet'")

con.sql(sys.argv[1]).show(max_rows=50)
