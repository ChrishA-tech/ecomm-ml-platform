import numpy as np, pandas as pd
H = 7  # forecast horizon: predict day t using history up to t-7; price/promo are known in advance
FEATURES = ["lag_7", "lag_14", "lag_28", "roll7", "roll28", "dow", "month", "price", "promo", "is_bf"]

def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sku", "date"]).copy()
    g = df.groupby("sku")["units"]
    df["lag_7"] = g.shift(H)
    df["lag_14"] = g.shift(H + 7)
    df["lag_28"] = g.shift(H + 21)
    df["roll7"] = g.transform(lambda s: s.shift(H).rolling(7).mean())
    df["roll28"] = g.transform(lambda s: s.shift(H).rolling(28).mean())
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_bf"] = ((df["date"].dt.month == 11) & (df["date"].dt.day >= 24) & (df["date"].dt.day <= 30)).astype(int)
    return df.dropna(subset=FEATURES)
