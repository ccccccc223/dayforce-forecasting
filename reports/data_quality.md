# Data quality report

Source: six Excel exports from the Dayforce SQL database (daily sales and special days for three retail clients). Figures below come from `data/processed/quality_log.json`, written by `python -m src.clean`. Only data from 2022-01-01 onward is used, because earlier sales are distorted by COVID-19 restrictions.

## Summary

| | Longo's (Canada, grocery) | Moto (UK, motorway services) | Costa (UK, coffee shops) |
|---|---|---|---|
| Sales KPI | KpiId 1 | KpiId 121 (adjusted gross sales) | KpiId 300 |
| Real data ends | 2024-09-22 | 2023-12-19 | 2024-08-25 |
| Stores used | 39 | 65 | 974 |
| Store-days used | 36,982 | 46,344 | 886,034 |
| Closed store-days | 1,451 | 14 | 12,258 |
| Spike anomalies | 0 | 9 | 2 |

## Issues found and how they are handled

### 1. Special days missing from the original merge (root cause in the extract SQL)
Special days are assigned to org units through `SpecialDayAssignment`, and an org unit can be a parent node such as the whole brand, not only a store. The original extract never expanded the org hierarchy down to stores, so a join on store id + date dropped every parent-level assignment.
- **Costa:** all 62 post-2022 special-day rows sit on parent unit 132436, so the original merged file had no special days at all.
- **Moto:** 1,792 rows are assigned to stores and 38 to the brand root (1000). Another 216 rows sit on 9 other non-store units (1510, 1512, 1538, 1555, 1568, 1570–1573) that cannot be mapped without the org hierarchy table, so they are left out.
- **Fix:** parent-level (brand root) assignments are applied to every store of that client. A data check confirms every Costa and Moto store has Christmas Day flagged.

### 2. Zero placeholder rows after the extract date
The KPI table holds pre-created zero rows for future dates, and the extract had no date upper bound. From a client-wide cutoff onward, every store shows 0. The cutoff is detected as the last day on which at least 90% of the usual share of stores reported sales. Rows dropped: Longo's 4,550, Moto 5,046, Costa 40,556.

### 3. Truncated Costa workbook
The Costa export has 1,046,486 rows, at Excel's row limit (1,048,576). The last store in the file stops mid-history on 2022-09-04, and any stores sorted after it are missing entirely. This cannot be recovered from the export. The affected store is treated like other stores that ended early.

### 4. Non-store org units (Longo's)
The Longo's extract includes a distribution centre, central kitchen, commissary, training store, café units and the online grocery service. These 8 units are excluded.

### 5. Stores that never traded or ended early
- Never traded (all zeros): Longo's 4, Moto 3, Costa 66. Excluded.
- Each store's data is limited to its trading period, from its first to its last positive sale. This drops pre-opening and post-closure zeros.
- Ended more than 28 days before the client cutoff: Moto 1, Costa 107. Kept for training, flagged `ended_early` in `stores.parquet`.

### 6. Closures, including mislabelled closures
The special-day `IsClosed` field is unreliable:
- Costa marks Christmas Day as open, yet about 94% of stores sell nothing that day.
- Longo's marks "Victoria Day Closed", yet only 21% of those store-days have zero sales.

Closures are therefore taken from the sales themselves. A day counts as closed when sales are 0 or below 5% of the store's centred 29-day median (a few stray transactions). The 5% rule reclassified 91 Longo's, 4 Moto and 2,155 Costa store-days as closed. Two Longo's stores close every weekend, and that pattern is kept.

### 7. Spikes
A day is a spike when sales are more than 4× the store's centred 29-day median. The main case is Moto store 1029: nine days in April 2022 ran at 17–65× normal, up to 1.13M against a typical ~15k. Spikes are flagged `is_anomaly` for exclusion from training and evaluation. Low-sales days are deliberately **not** flagged, because most of them are holidays (Christmas, Easter Sunday, Boxing Day, the 2022-09-19 state funeral, Ontario Family Day) and the model needs to learn those effects.

### 8. Smaller issues
- Costa: 5 negative sales values (refunds exceeding sales), clipped to 0.
- Longo's: duplicated store-days, but only on the 2025-01-07 placeholder date.
- Longo's: `SpecialDayLongName` contains dates that Excel auto-converted from text. `SpecialDayShortName` is used instead.
- Longo's: the flag column `D` is undocumented and only marks one non-trading unit, so it is dropped.
- Moto: the "merged" file is exactly the history file filtered to 2022+, so only the history file is used.
- Special-day coverage is uneven. Longo's has only a few Ontario holidays (no Christmas or New Year), and Moto's table ends in December 2023. Costa's table uses England's bank-holiday dates for every store, but 85 of its stores (and 2 Moto stores) are in Scotland, where the summer bank holiday falls on a different date and 2 January and St Andrew's Day are also holidays. `src/holiday_calendar.py` therefore builds an official calendar per region (Ontario, England, Wales, Scotland), plus fixed retail events (Mother's Day, Easter Sunday, Black Friday, Christmas Eve and others). Each store's region comes from its postcode.

## Weather data
- **Longo's:** included in the export (Open-Meteo, per postal code). Its 760 missing cells all fall in January 2025, after the real data ends, so none remain after merging.
- **Moto and Costa:** fetched by `src/geocode_uk.py` and `src/weather_uk.py`. The 997 postcodes were geocoded with postcodes.io: 961 exact, 32 retired postcodes via the terminated-postcode lookup, and 4 by outward code. Daily weather for 2022-01-01 to 2024-08-25 comes from the Open-Meteo historical archive with the same 8 variables as Longo's.
- **Grouping:** the free API allows 10,000 weighted calls per day, and a 3-year daily request costs about 70 calls per location. Stores are therefore grouped into 0.5° grid cells (115 cells, up to 106 stores each). Each cell is queried at the mean position of its stores. The median store is 11 km from its query point and the furthest is 42 km. Daily temperature varies little at this scale, but local rainfall can differ.
- **Result:** 111,320 rows with no missing values. Sanity check: the London cell peaks at 38.0 °C on 2022-07-19, the record heatwave day.

## Consequences for modelling
- Moto has under 2 years of usable history, which is thin for yearly seasonality.
- Costa dominates by store count, so metrics are reported per client.
- Closed days are known in advance in practice (scheduled closures), so models are evaluated on trading days only.
