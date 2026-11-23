"""Train LightGBM quantile forecasters (q50, q90), backtest on the last 12 weeks vs baselines,
log to MLflow, save versioned artifacts + drift reference."""
import os; os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true"); os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import json, sys, numpy as np, pandas as pd, lightgbm as lgb
from features import build, FEATURES

def wape(y, p): return float(np.abs(y - p).sum() / max(y.sum(), 1e-9))
def pinball(y, p, q): d = y - p; return float(np.mean(np.maximum(q * d, (q - 1) * d)))

def train(data_path, out_dir, log_mlflow=True):
    df = build(pd.read_parquet(data_path))
    cut_test = df["date"].max() - pd.Timedelta(days=83)
    cut_val = cut_test - pd.Timedelta(days=84)
    tr, va, te = df[df.date < cut_val], df[(df.date >= cut_val) & (df.date < cut_test)], df[df.date >= cut_test]
    models, params = {}, dict(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=7)
    for q in (0.5, 0.9):
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
        m.fit(tr[FEATURES], tr["units"], eval_set=[(va[FEATURES], va["units"])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        models[q] = m
    y = te["units"].values
    p50, p90 = models[0.5].predict(te[FEATURES]), models[0.9].predict(te[FEATURES])
    base_naive, base_roll = te["lag_7"].values, te["roll28"].values
    # paired bootstrap over SKUs for WAPE difference (model - seasonal naive)
    rng = np.random.default_rng(7); skus = te["sku"].unique(); diffs = []
    tmp = te.assign(ae_m=np.abs(te["units"].values - p50), ae_n=np.abs(te["units"].values - base_naive))
    agg = tmp.groupby("sku").agg(ae_m=("ae_m", "sum"), ae_n=("ae_n", "sum"), y=("units", "sum"))
    for _ in range(1000):
        s = agg.loc[rng.choice(skus, len(skus))]
        diffs.append((s.ae_m.sum() - s.ae_n.sum()) / s.y.sum())
    metrics = {
        "data": "synthetic", "rows_train": len(tr), "rows_test": len(te), "test_weeks": 12, "skus": int(len(skus)),
        "wape_model": wape(y, p50), "wape_seasonal_naive": wape(y, base_naive), "wape_roll28": wape(y, base_roll),
        "wape_diff_vs_naive_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "q90_coverage": float((y <= p90).mean()), "q90_pinball": pinball(y, p90, 0.9),
        "best_iter_q50": int(models[0.5].best_iteration_ or params["n_estimators"]),
    }
    os.makedirs(out_dir, exist_ok=True)
    models[0.5].booster_.save_model(f"{out_dir}/q50.txt"); models[0.9].booster_.save_model(f"{out_dir}/q90.txt")
    ref = {f: tr[f].astype(float).tolist()[::50] for f in ["price", "promo", "roll7", "roll28"]}
    json.dump(ref, open(f"{out_dir}/drift_reference.json", "w"))
    json.dump(metrics, open(f"{out_dir}/metrics.json", "w"), indent=2)
    if log_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(f"file:{os.path.dirname(os.path.dirname(os.path.abspath(out_dir)))}/mlruns"); mlflow.set_experiment("ecomm-demand")
            with mlflow.start_run():
                mlflow.log_params({k: v for k, v in params.items() if k != "verbose"})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
                [mlflow.log_artifact(f"{out_dir}/{f}") for f in ("q50.txt", "q90.txt", "metrics.json", "drift_reference.json")]
        except Exception as e: print("mlflow skipped:", e)
    return metrics

if __name__ == "__main__":
    print(json.dumps(train(sys.argv[1], sys.argv[2]), indent=2))
