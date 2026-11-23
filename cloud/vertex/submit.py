"""Submit the compiled pipeline to Vertex AI. Requires a GCP project + auth; NOT run in this repo."""
import sys
from google.cloud import aiplatform
project, bucket = sys.argv[1], sys.argv[2]
aiplatform.init(project=project, location="europe-west2", staging_bucket=f"gs://{bucket}")
aiplatform.PipelineJob(display_name="ecomm-demand", template_path="cloud/vertex/pipeline.yaml",
    parameter_values={"data_uri": f"gs://{bucket}/orders.parquet", "project": project}).run()
