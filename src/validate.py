"""Data validation gate run before training (fails the pipeline on bad data)."""
import pandas as pd

def validate(df: pd.DataFrame) -> dict:
    checks = {
        "no_missing": int(df[["date", "sku", "price", "promo", "units"]].isna().sum().sum()) == 0,
        "no_dupe_keys": not df.duplicated(["sku", "date"]).any(),
        "units_non_negative": bool((df["units"] >= 0).all()),
        "price_positive": bool((df["price"] > 0).all()),
        "promo_binary": bool(df["promo"].isin([0, 1]).all()),
    }
    checks["passed"] = all(checks.values())
    return checks

if __name__ == "__main__":
    import sys, json
    r = validate(pd.read_parquet(sys.argv[1]))
    print(json.dumps(r))
    sys.exit(0 if r["passed"] else 1)
