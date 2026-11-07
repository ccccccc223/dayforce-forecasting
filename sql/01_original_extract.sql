-- Original extraction queries (2025 capstone), run on the client's internal database.
-- Table and column names are generalised; the logic is unchanged.
-- Known problems are listed at the bottom and fixed in 02_fixed_extract.sql.

-- Daily sales per org unit. KpiId differs by client (Longo's 1, Moto 121, Costa 300).
SELECT
    a.org_unit_id,
    a.day_id,
    a.kpi_id,
    a.value,
    a.client_id,
    o.long_name,
    o.address,
    o.postal_code
FROM kpi_actual_daily AS a
JOIN org_unit AS o
    ON a.org_unit_id = o.org_unit_id
WHERE a.kpi_id = 1;

-- Special days with their dates and the org units they are assigned to.
SELECT
    s.special_day_id,
    d.day_date,
    s.short_name,
    s.long_name,
    s.client_id,
    d.is_closed,
    a.org_unit_id
FROM special_day AS s
JOIN special_day_date AS d
    ON s.special_day_id = d.special_day_id
JOIN special_day_assignment AS a
    ON s.special_day_id = a.special_day_id;

-- Problems found later:
-- 1. special_day_assignment.org_unit_id can be a parent node (a brand or region), not a
--    store. Joining the second result to sales on store id + date silently drops those
--    rows: every Costa special day was lost this way.
-- 2. No upper date bound: the KPI table holds pre-created zero rows for future dates,
--    which were exported as if they were real zero-sales days.
-- 3. No lower date bound: pre-2022 (COVID-affected) data was exported and filtered later.
-- 4. Non-store org units (distribution centres, kitchens, online channel) are not
--    filtered out.
