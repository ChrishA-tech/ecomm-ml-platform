"""Simulate a drifted production week and show the monitoring decision."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "src")
from gen_data import make
from features import build, FEATURES
from monitor import psi, decide
import lightgbm as lgb
ref = json.load(open("artifacts/model_v1/drift_reference.json")); m = json.load(open("artifacts/model_v1/metrics.json"))
df = make(seed=7).query("date >= '2024-10-01'").copy()            # same process, held-out window
d2 = df.copy(); d2["price"] *= 1.25; d2["promo"] = (np.random.default_rng(3).random(len(d2)) < 0.25).astype(int)  # drift
b = lgb.Booster(model_file="artifacts/model_v1/q50.txt")
def run(frame, label):
    full = make(seed=7); full = full.merge(frame[["sku", "date", "price", "promo"]], on=["sku", "date"], how="left", suffixes=("", "_n"))
    full["price"] = full["price_n"].fillna(full["price"]); full["promo"] = full["promo_n"].fillna(full["promo"]).astype(int)
    f = build(full.drop(columns=["price_n", "promo_n"])); f = f[f.date >= "2024-10-01"]
    p = b.predict(f[FEATURES]); wape = float(np.abs(f.units - p).sum() / f.units.sum())
    psis = {k: psi(ref[k], f[k].astype(float).tolist()) for k in ref}
    print(label, "PSI:", {k: round(v, 3) for k, v in psis.items()}, "WAPE:", round(wape, 3),
          decide(psis, wape, m["wape_model"]))
run(df, "stable ")
run(d2, "drifted")
