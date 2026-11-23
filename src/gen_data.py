"""Synthetic ecommerce daily demand: 100 SKUs x 730 days with weekly/annual seasonality,
promotions, price elasticity and a Black Friday spike. SYNTHETIC - labelled as such everywhere."""
import numpy as np, pandas as pd, sys

def make(n_sku=100, days=730, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days)
    out = []
    for s in range(n_sku):
        base = rng.gamma(3, 4) + 2
        p0 = rng.uniform(5, 60)
        dow = rng.normal(1, 0.15, 7)
        phase = rng.uniform(0, 2 * np.pi)
        trend = rng.normal(0.0002, 0.0003)
        elast = rng.uniform(0.8, 2.2)
        promo = (rng.random(days) < 0.08).astype(int)
        price = p0 * (1 - 0.2 * promo) * (1 + rng.normal(0, 0.01, days))
        doy = dates.dayofyear.values
        bf = ((dates.month == 11) & (dates.day >= 24) & (dates.day <= 30)).astype(int)
        lam = (base * dow[dates.dayofweek.values]
               * (1 + 0.3 * np.sin(2 * np.pi * doy / 365 + phase))
               * (1 + trend * np.arange(days))
               * (price / p0) ** (-elast)
               * (1 + 0.8 * bf))
        out.append(pd.DataFrame({"date": dates, "sku": f"SKU{s:03d}", "price": price.round(2),
                                 "promo": promo, "units": rng.poisson(lam)}))
    return pd.concat(out, ignore_index=True)

if __name__ == "__main__":
    df = make()
    df.to_parquet(sys.argv[1] if len(sys.argv) > 1 else "artifacts/orders.parquet", index=False)
    print(len(df), "rows (synthetic)")
