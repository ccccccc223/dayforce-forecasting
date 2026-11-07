"""Paths and project-wide constants."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Pre-2022 sales are distorted by COVID-19 restrictions.
START_DATE = pd.Timestamp("2022-01-01")

# Raw exports -> cached parquet name (one parquet per sheet that holds data).
RAW_FILES = {
    "longos_sales_weather": ("Longos_Sales_and_Weather_Data.xlsx", "Merged_Sales_and_Weather_Data"),
    "costa_sales": ("Merged_Total_Sales_and_Special_Days_Costa.xlsx", "Sheet1"),
    "costa_specialdays": ("Specialday(costa.xlsx", "Sheet1"),
    "moto_sales_merged": ("Merged_Total_Sales_and_Special_Days(Moto).xlsx", "Sheet1"),
    "moto_sales_history": ("Past adjusted gross sales amount with postal code (Moto).xlsx", "Sheet1"),
    "moto_specialdays": ("Specialday(Moto).xlsx", "Sheet1"),
}

# KpiId used for daily sales in each client's extract.
SALES_KPI = {"longos": 1, "moto": 121, "costa": 300}
