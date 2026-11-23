"""Vertex AI Pipeline (KFP v2): validate -> train -> evaluate gate -> register.
Compiled to pipeline.yaml locally. NOT executed on a live GCP project from this repo."""
from typing import NamedTuple
from kfp import dsl, compiler

BASE = "python:3.11-slim"

@dsl.component(base_image=BASE, packages_to_install=["pandas", "pyarrow"])
def validate_op(data_uri: str) -> bool:
    import pandas as pd
    df = pd.read_parquet(data_uri)
    ok = (not df.duplicated(["sku", "date"]).any()) and bool((df.units >= 0).all()) and bool((df.price > 0).all())
    if not ok: raise ValueError("data validation failed")
    return ok

@dsl.component(base_image=BASE, packages_to_install=["pandas", "pyarrow", "lightgbm", "numpy"])
def train_op(data_uri: str, model: dsl.Output[dsl.Model], metrics: dsl.Output[dsl.Metrics]) -> NamedTuple(
        "TrainOut", [("wape_model", float), ("wape_naive", float)]):
    import subprocess, json, sys
    # container image in production bakes src/ in; here we import the project package from the image
    sys.path.insert(0, "/srv/src")
    from train import train
    m = train(data_uri, model.path, log_mlflow=False)
    metrics.log_metric("wape_model", m["wape_model"]); metrics.log_metric("wape_naive", m["wape_seasonal_naive"])
    model.metadata["wape_model"] = m["wape_model"]
    from collections import namedtuple
    return namedtuple("TrainOut", ["wape_model", "wape_naive"])(m["wape_model"], m["wape_seasonal_naive"])

@dsl.component(base_image=BASE)
def gate_op(wape_model: float, wape_naive: float, max_ratio: float) -> bool:
    """True only if the model beats the seasonal-naive baseline by the required margin."""
    return wape_model <= wape_naive * max_ratio

@dsl.component(base_image=BASE, packages_to_install=["google-cloud-aiplatform"])
def register_op(model: dsl.Input[dsl.Model], project: str, region: str, display_name: str):
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)
    aiplatform.Model.upload(display_name=display_name, artifact_uri=model.uri,
                            serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest")

@dsl.pipeline(name="ecomm-demand-forecast", description="Validate, train, gate and register demand models")
def pipeline(data_uri: str, project: str, region: str = "europe-west2", display_name: str = "ecomm-demand",
             max_ratio: float = 0.9):
    v = validate_op(data_uri=data_uri)
    t = train_op(data_uri=data_uri).after(v)
    g = gate_op(wape_model=t.outputs["wape_model"], wape_naive=t.outputs["wape_naive"], max_ratio=max_ratio)
    with dsl.If(g.output == True, name="beats-baseline"):  # noqa: E712
        register_op(model=t.outputs["model"], project=project, region=region, display_name=display_name)

if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "cloud/vertex/pipeline.yaml")
    print("compiled cloud/vertex/pipeline.yaml")
