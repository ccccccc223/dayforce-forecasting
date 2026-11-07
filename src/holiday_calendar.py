"""Official public holidays and fixed retail events per region (ON, ENG, WLS, SCT).

The clients' own special-day tables are incomplete (Longo's has no Christmas or New
Year; Moto's ends in December 2023; Costa's uses England's dates for Scottish stores),
so this calendar is the consistent baseline for holiday features.

Output: data/processed/holidays.parquet  (region, date, name, kind = public | event)
Run: python -m src.holiday_calendar
"""
import holidays
import pandas as pd
from dateutil.easter import easter
from holidays.constants import OPTIONAL, PUBLIC

from src.config import PROCESSED

YEARS = range(2022, 2026)
REGIONS = {"ON": ("CA", "ON"), "ENG": ("GB", "ENG"), "WLS": ("GB", "WLS"), "SCT": ("GB", "SCT")}


def nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """n-th given weekday (Mon=0) of a month."""
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def public_holidays(region: str) -> list[dict]:
    country, subdiv = REGIONS[region]
    cal = dict(holidays.country_holidays(country, subdiv=subdiv, years=YEARS, categories=PUBLIC))
    if region == "ON":
        # Civic Holiday is not statutory but is widely observed by Ontario retail.
        optional = holidays.country_holidays(country, subdiv=subdiv, years=YEARS, categories=OPTIONAL)
        cal |= {day: name for day, name in optional.items() if name == "Civic Holiday"}
    return [{"region": region, "date": pd.Timestamp(day), "name": name, "kind": "public"}
            for day, name in cal.items()]


def retail_events(region: str) -> list[dict]:
    rows = []
    for y in YEARS:
        e = pd.Timestamp(easter(y))
        mothers = e - pd.Timedelta(days=21) if region != "ON" else nth_weekday(y, 5, 6, 2)
        black_friday = nth_weekday(y, 11, 3, 4) + pd.Timedelta(days=1)
        for date, name in [
            (pd.Timestamp(y, 2, 14), "Valentine's Day"),
            (mothers, "Mother's Day"),
            (e, "Easter Sunday"),
            (nth_weekday(y, 6, 6, 3), "Father's Day"),
            (black_friday, "Black Friday"),
            (pd.Timestamp(y, 12, 24), "Christmas Eve"),
            (pd.Timestamp(y, 12, 31), "New Year's Eve"),
        ]:
            rows.append({"region": region, "date": date, "name": name, "kind": "event"})
    return rows


def main() -> None:
    rows = []
    for region in REGIONS:
        rows += public_holidays(region) + retail_events(region)
    cal = pd.DataFrame(rows).sort_values(["region", "date"]).reset_index(drop=True)
    cal.to_parquet(PROCESSED / "holidays.parquet", index=False)
    print(cal.groupby(["region", "kind"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
