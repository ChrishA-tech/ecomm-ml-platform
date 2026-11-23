# ecomm-ml-platform: demand forecasting, deployed and monitored (Vertex AI / Azure ML / AWS ready)

End-to-end ML lifecycle for ecommerce demand: **validate -> train (LightGBM quantile) -> backtest -> serve (FastAPI/Docker) -> monitor (PSI + WAPE) -> retrain decision**, with the cloud packaging for **Google Vertex AI**, **Azure ML** and **AWS S3**.

> **Data is synthetic** (100 SKUs x 730 days, weekly/annual seasonality, promotions, price elasticity, Black Friday). Results show the pipeline works; they are not claims about a real business.

## Results (12-week hold-out, 8,400 SKU-days)
| Metric | Value |
|---|---|
| WAPE, LightGBM q50 | **22.1%** |
| WAPE, seasonal naive (same day last week) | 36.4% |
| WAPE, 28-day moving average | 29.9% |
| WAPE difference vs naive, paired bootstrap over SKUs | -14.9 to -13.6 pts (95% CI, excludes 0) |
| q90 coverage (target 90%) | 89.7% |
| API under load (1 uvicorn worker, sandbox, 16 threads, 20 rows/request) | 111.6 req/s, p50 136 ms, p95 234 ms, 0 errors |
| Drift monitor, stable week | PSI price 0.003, no retrain |
| Drift monitor, drifted week (price +25%, promo 8%->25%) | PSI price 0.394, promo 0.221, WAPE 22.2% -> 27.5%, **retrain triggered** |

## What is verified vs not
| Component | Status |
|---|---|
| Data validation gate, training, backtest, MLflow experiment tracking (file store), FastAPI, monitoring | **Run locally, 6 pytest tests pass** |
| Vertex AI pipeline (KFP v2: validate, train, metric gate, register) | **Compiled** to `pipeline.yaml`; test checks the components and the condition. **Not executed on a GCP project.** |
| Azure ML command job, managed online endpoint, deployment | **Parsed and schema-validated** with the `azure-ai-ml` SDK loaders. **Not deployed to a workspace.** |
| AWS S3 versioned model registry (push / promote / pull latest) | **Tested against moto**, not real AWS |
| Dockerfile (non-root user) | Written; **not built in this sandbox** (no Docker daemon) |

## Run
```bash
pip install -r requirements.txt pytest httpx moto boto3 kfp azure-ai-ml pyyaml mlflow-skinny
python src/gen_data.py artifacts/orders.parquet && python src/validate.py artifacts/orders.parquet
cd src && python train.py ../artifacts/orders.parquet ../artifacts/model_v1 && cd ..
python -m pytest -q tests
python scripts/drift_demo.py
MODEL_DIR=artifacts/model_v1 uvicorn app.main:app --port 8080 & python scripts/loadtest.py
# Cloud (needs your accounts):
python cloud/vertex/submit.py <gcp-project> <bucket>
az ml job create -f cloud/azureml/train_job.yml
```
