import os
import json
from google.oauth2.sts import Client
import tensorflow as tf
from google.cloud import storage

from typing import Any
from google.api_core.page_iterator import Iterator
from io import StringIO
import pandas as pd

def gcs_jsonl_to_tfrecord_all_files(gcs_jsonl_path: str, gcs_tfrecord_prefix: str) -> None:
    for blob in _list_blobs(gcs_jsonl_path):
        jsonl_text: str = blob.download_as_string().decode('utf-8')
        jsonl_to_tfrecord(jsonl_text, _output_path(blob.name, gcs_tfrecord_prefix))


def jsonl_to_tfrecord(jsonl_text: str, output_path: str) -> None:
    if _gcs_path_exists(output_path):
        print(f"Skipping {output_path} - already exists")
        return

    with tf.io.TFRecordWriter(output_path) as writer:
        rows: list[dict[str, str]] = pd.read_json(StringIO(jsonl_text), lines=True).to_dict(orient='records')
        for row in rows:
            features: dict[str, tf.train.Feature] = {
                key: tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))
                for key, value in row.items()
            }

            # Create an Example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=features))

            # Write the serialized example to the TFRecord file
            writer.write(example.SerializeToString())
            print(".", end="")
    print(f'Finished {output_path}')

def _split_bucket_and_path(gcs_path: str) -> tuple[str, str]:
    bucket_name, path = gcs_path.replace('gs://', '').split('/', maxsplit=1)
    return bucket_name, path

def _list_blobs(path: str) -> Iterator:
    client = storage.Client()
    bucket_name, gcs_jsonl_prefix = _split_bucket_and_path(path)
    bucket: storage.Bucket = client.get_bucket(bucket_name)
    return bucket.list_blobs(prefix=gcs_jsonl_prefix)

def _output_path(jsonl_file_path: str, gcs_tfrecord_prefix: str) -> str:
    jsonl_file_name: str = jsonl_file_path.split('/')[-1]
    tfrecord_file_name: str = jsonl_file_name.replace('.jsonl', '.tfrecords')
    return os.path.join(gcs_tfrecord_prefix, tfrecord_file_name)

def _gcs_path_exists(path: str) -> bool:
    client = storage.Client()
    bucket_name, path = _split_bucket_and_path(path)
    bucket: storage.Bucket = client.get_bucket(bucket_name)
    return bucket.blob(path).exists()

gcs_jsonl_bucket = 'mazumdera-test-bucket'
prefix = 'lg/jsonl-data'
tfrecord_dir = 'gs://'+gcs_jsonl_bucket+'/lg/tfrecord-data-carter/' 

gcs_jsonl_to_tfrecord_all_files(gcs_jsonl_path=f"gs://{gcs_jsonl_bucket}/{prefix}", gcs_tfrecord_prefix=tfrecord_dir)
# jsonl_to_tfrecord_all_files(gcs_jsonl_bucket, prefix, tfrecord_dir) 


