"""Offline schema validation of the Azure ML specs with the azure-ai-ml SDK loaders (no workspace needed)."""
from azure.ai.ml import load_job, load_online_endpoint, load_online_deployment
j = load_job("cloud/azureml/train_job.yml"); e = load_online_endpoint("cloud/azureml/endpoint.yml")
d = load_online_deployment("cloud/azureml/deployment.yml")
print("job:", j.type, "| endpoint:", e.name, e.auth_mode, "| deployment:", d.name, d.instance_type, d.instance_count)
