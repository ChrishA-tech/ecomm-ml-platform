import httpx, time, statistics as st, concurrent.futures as cf, json
row = dict(lag_7=10, lag_14=9, lag_28=11, roll7=10, roll28=10, dow=2, month=5, price=20, promo=0, is_bf=0)
def one(_):
    t = time.perf_counter(); r = httpx.post("http://127.0.0.1:8080/predict", json={"rows": [row] * 20}, timeout=10)
    return time.perf_counter() - t, r.status_code
N = 2000; t0 = time.perf_counter()
with cf.ThreadPoolExecutor(16) as ex: res = list(ex.map(one, range(N)))
el = time.perf_counter() - t0; lat = sorted(x[0] * 1000 for x in res)
print(json.dumps({"requests": N, "batch_rows": 20, "concurrency": 16, "errors": sum(x[1] != 200 for x in res),
  "rps": round(N / el, 1), "p50_ms": round(st.median(lat), 1), "p95_ms": round(lat[int(.95 * N)], 1)}))
