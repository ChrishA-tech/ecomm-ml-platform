"""Versioned model registry on S3: upload artifacts under models/<name>/<version>/, promote a 'latest' pointer,
and fetch the pointed version. Tested against moto; set AWS creds + bucket to use real S3."""
import json, os, boto3

def push(bucket, name, version, src_dir, s3=None):
    s3 = s3 or boto3.client("s3")
    for f in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, f)
        if os.path.isfile(p): s3.upload_file(p, bucket, f"models/{name}/{version}/{f}")
    return f"s3://{bucket}/models/{name}/{version}/"

def promote(bucket, name, version, s3=None):
    s3 = s3 or boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=f"models/{name}/latest.json", Body=json.dumps({"version": version}))

def pull_latest(bucket, name, dst, s3=None):
    s3 = s3 or boto3.client("s3")
    v = json.loads(s3.get_object(Bucket=bucket, Key=f"models/{name}/latest.json")["Body"].read())["version"]
    os.makedirs(dst, exist_ok=True)
    pre = f"models/{name}/{v}/"
    for o in s3.list_objects_v2(Bucket=bucket, Prefix=pre).get("Contents", []):
        s3.download_file(bucket, o["Key"], os.path.join(dst, os.path.basename(o["Key"])))
    return v
