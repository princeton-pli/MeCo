import logging
import os
from typing import List
from urllib.parse import urlparse

import boto3
import smart_open
from tqdm import tqdm

logger = logging.getLogger(__name__)

S3_CACHE_DIR = os.environ.get("S3_CACHE_DIR") or os.path.expanduser("~/.cache/oe_eval_s3_cache")


def cache_s3_folder(s3_path, cache_dir=None):
    cache_dir = cache_dir or S3_CACHE_DIR
    parsed_url = urlparse(s3_path)
    bucket_name = parsed_url.netloc
    s3_folder = parsed_url.path.lstrip("/")
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    files: List[str] = []
    s3_filenames = []
    for obj in bucket.objects.filter(Prefix=s3_folder):
        s3_filenames.append(obj.key)
    logger.info(f"Downloading {len(s3_filenames)} files from {s3_path} to {cache_dir}")
    for s3_filename in s3_filenames:
        local_filename = os.path.join(cache_dir, s3_filename)
        files.append(local_filename)
        if os.path.exists(local_filename):
            continue
        os.makedirs(os.path.dirname(local_filename), exist_ok=True)
        bucket.download_file(s3_filename, local_filename)
    local_dir = sorted([os.path.dirname(file) for file in files], key=len)[0]
    logger.info(f"Finished downloading to {local_dir}")
    return {"local_dir": local_dir, "files": files}


def upload_directory(local_dir: str, remote_dir: str):
    local_paths = [
        os.path.join(root, post_fn) for root, _, files in os.walk(local_dir) for post_fn in files
    ]
    dest_paths = [
        f"{remote_dir.rstrip('/')}/{os.path.relpath(local_path, local_dir).lstrip('/')}"
        for local_path in local_paths
    ]
    it = tqdm(
        iterable=zip(local_paths, dest_paths),
        total=len(local_paths),
        desc=f"Uploading files from {local_dir} to {remote_dir}",
    )
    for local_path, dest_path in it:
        with smart_open.open(local_path, "rb") as f, smart_open.open(dest_path, "wb") as g:
            g.write(f.read())
