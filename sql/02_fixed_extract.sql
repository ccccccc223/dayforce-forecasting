-- Fixed extraction (SQL Server dialect). Written after the fact from the issues found in
-- the exported data; it has NOT been run against the source database, and the org-unit
-- hierarchy column (parent_org_unit_id) and store-type column (org_unit_type) are assumed.

DECLARE @kpi_id     INT  = 300;           -- client's daily sales KPI
DECLARE @start_date DATE = '2022-01-01';  -- exclude COVID-affected history
DECLARE @end_date   DATE = '2024-08-25';  -- extract date: later rows are zero placeholders

-- Daily sales for stores only, inside the date window.
SELECT
    a.org_unit_id,
    a.day_id,
    a.value,
    a.client_id,
    o.long_name,
    o.postal_code
FROM kpi_actual_daily AS a
JOIN org_unit AS o
    ON a.org_unit_id = o.org_unit_id
WHERE a.kpi_id = @kpi_id
  AND a.day_id BETWEEN @start_date AND @end_date
  AND o.org_unit_type = 'Store';

-- Special days resolved to stores: an assignment to any ancestor node applies to every
-- store beneath it.
WITH org_tree AS (
    -- each store is its own descendant ...
    SELECT org_unit_id AS ancestor_id, org_unit_id AS store_id, parent_org_unit_id
    FROM org_unit
    WHERE org_unit_type = 'Store'
    UNION ALL
    -- ... and a descendant of every node above it
    SELECT p.org_unit_id, t.store_id, p.parent_org_unit_id
    FROM org_tree AS t
    JOIN org_unit AS p
        ON p.org_unit_id = t.parent_org_unit_id
)
SELECT DISTINCT
    t.store_id,
    d.day_date,
    s.special_day_id,
    s.short_name
FROM special_day_assignment AS a
JOIN org_tree AS t
    ON t.ancestor_id = a.org_unit_id
JOIN special_day AS s
    ON s.special_day_id = a.special_day_id
JOIN special_day_date AS d
    ON d.special_day_id = s.special_day_id
WHERE d.day_date BETWEEN @start_date AND @end_date
OPTION (MAXRECURSION 20);
-- is_closed is deliberately not exported: it proved unreliable (e.g. Christmas marked
-- open while ~94% of Costa stores sold nothing). Closures are derived from sales instead.
