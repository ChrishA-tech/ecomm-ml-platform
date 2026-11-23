"""Production monitoring: PSI feature drift + rolling WAPE decay -> retrain decision."""
import json, sys, numpy as np, pandas as pd

def psi(ref, cur, bins=10):
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: edges = np.array([ref.min() - 1e-9, ref.mean(), ref.max() + 1e-9])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref); c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))

def decide(psis: dict, wape_now: float, wape_ref: float, psi_thr=0.2, wape_tol=0.10):
    drifted = [k for k, v in psis.items() if v > psi_thr]
    degraded = wape_now > wape_ref * (1 + wape_tol)
    return {"drifted_features": drifted, "wape_degraded": bool(degraded), "retrain": bool(drifted or degraded)}

if __name__ == "__main__":
    ref = json.load(open(sys.argv[1])); cur = json.load(open(sys.argv[2]))
    p = {k: psi(ref[k], cur[k]) for k in ref}
    print(json.dumps({"psi": p}, indent=2))
