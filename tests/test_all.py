import sys, json, os, pandas as pd, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from fastapi.testclient import TestClient

def test_validation_gate_catches_bad_data():
    from validate import validate
    df = pd.read_parquet("artifacts/orders.parquet")
    assert validate(df)["passed"]
    bad = pd.concat([df, df.iloc[:5]]); assert not validate(bad)["no_dupe_keys"]
    neg = df.copy(); neg.loc[0, "units"] = -1; assert not validate(neg)["units_non_negative"]

def test_model_beats_baselines_with_ci_excluding_zero():
    m = json.load(open("artifacts/model_v1/metrics.json"))
    assert m["wape_model"] < m["wape_seasonal_naive"] and m["wape_model"] < m["wape_roll28"]
    lo, hi = m["wape_diff_vs_naive_ci95"]; assert hi < 0
    assert 0.85 <= m["q90_coverage"] <= 0.95

def test_api_predict_and_validation():
    from app.main import app
    c = TestClient(app); assert c.get("/health").json()["status"] == "ok"
    row = dict(lag_7=10, lag_14=9, lag_28=11, roll7=10, roll28=10, dow=2, month=5, price=20, promo=0, is_bf=0)
    r = c.post("/predict", json={"rows": [row, {**row, "promo": 1, "price": 16}]}).json()
    assert len(r["q50"]) == 2 and all(h >= l for l, h in zip(r["q50"], r["q90"])) and r["q50"][1] > r["q50"][0]
    assert c.post("/predict", json={"rows": [{**row, "price": -1}]}).status_code == 422
    assert c.post("/predict", json={"rows": []}).status_code == 422
    assert "predict_requests_total" in c.get("/metrics").text

def test_psi_and_retrain_decision():
    from monitor import psi, decide
    rng = np.random.default_rng(0); a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05 and psi(a, rng.normal(1, 1, 5000)) > 0.2
    assert decide({"x": 0.01}, 0.22, 0.22)["retrain"] is False
    assert decide({"x": 0.3}, 0.22, 0.22)["retrain"] is True and decide({"x": 0.0}, 0.30, 0.22)["wape_degraded"]

def test_s3_registry_roundtrip_with_moto(tmp_path):
    import boto3; from moto import mock_aws
    from cloud.aws import s3_registry as r
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-2")
        s3.create_bucket(Bucket="ecomm-models-test", CreateBucketConfiguration={"LocationConstraint": "eu-west-2"})
        r.push("ecomm-models-test", "demand", "v1", "artifacts/model_v1", s3); r.promote("ecomm-models-test", "demand", "v1", s3)
        assert r.pull_latest("ecomm-models-test", "demand", str(tmp_path), s3) == "v1"
        assert (tmp_path / "q50.txt").exists() and (tmp_path / "metrics.json").exists()

def test_vertex_pipeline_compiles_with_gate_and_azure_specs_load():
    import yaml, subprocess
    subprocess.run([sys.executable, "cloud/vertex/pipeline.py"], check=True, capture_output=True)
    spec = yaml.safe_load(open("cloud/vertex/pipeline.yaml"))
    comps = set(spec["components"]); assert {"comp-validate-op", "comp-train-op", "comp-gate-op", "comp-register-op"} <= comps
    assert any("condition" in c for c in spec["components"])
    from azure.ai.ml import load_job, load_online_endpoint, load_online_deployment
    assert load_job("cloud/azureml/train_job.yml").type == "command"
    assert load_online_deployment("cloud/azureml/deployment.yml").instance_count == 2
