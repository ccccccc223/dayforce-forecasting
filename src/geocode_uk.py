"""Look up latitude/longitude for UK store postcodes via postcodes.io.

Postcodes no longer in use fall back to the terminated-postcode lookup, then to the
outward code (e.g. "OX27"). Output: data/interim/uk_postcodes.parquet
Run: python -m src.geocode_uk
"""
import requests
import pandas as pd

from src.config import INTERIM, PROCESSED

API = "https://api.postcodes.io"


def lookup_bulk(postcodes: list[str]) -> dict[str, tuple[float, float]]:
    found = {}
    for i in range(0, len(postcodes), 100):  # API maximum per request
        batch = postcodes[i:i + 100]
        r = requests.post(f"{API}/postcodes", json={"postcodes": batch}, timeout=30)
        r.raise_for_status()
        for item in r.json()["result"]:
            res = item["result"]
            if res and res["latitude"] is not None:
                found[item["query"]] = (res["latitude"], res["longitude"])
    return found


def lookup_fallback(postcode: str) -> tuple[float, float, str] | None:
    r = requests.get(f"{API}/terminated_postcodes/{postcode}", timeout=30)
    if r.ok and r.json()["result"]["latitude"] is not None:
        res = r.json()["result"]
        return res["latitude"], res["longitude"], "terminated"
    outward = postcode[:-3]
    r = requests.get(f"{API}/outcodes/{outward}", timeout=30)
    if r.ok and r.json()["result"]["latitude"] is not None:
        res = r.json()["result"]
        return res["latitude"], res["longitude"], "outcode"
    return None


def main() -> None:
    stores = pd.read_parquet(PROCESSED / "stores.parquet")
    uk = stores[stores.client.isin(["moto", "costa"])]
    postcodes = sorted(uk.postal_code.str.upper().str.replace(" ", "").unique())

    found = lookup_bulk(postcodes)
    rows = [{"postal_code": pc, "lat": lat, "lon": lon, "match": "exact"} for pc, (lat, lon) in found.items()]
    for pc in sorted(set(postcodes) - set(found)):
        hit = lookup_fallback(pc)
        rows.append({"postal_code": pc, "lat": hit[0] if hit else None,
                     "lon": hit[1] if hit else None, "match": hit[2] if hit else "not_found"})

    out = pd.DataFrame(rows).sort_values("postal_code")
    out.to_parquet(INTERIM / "uk_postcodes.parquet", index=False)
    print(out.match.value_counts().to_string())
    print(out[out.match == "not_found"].to_string())


if __name__ == "__main__":
    main()
