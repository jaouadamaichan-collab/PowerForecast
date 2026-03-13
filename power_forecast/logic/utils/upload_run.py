"""
Utility function to upload ML runs to Google Cloud Storage.
"""

import os
from google.cloud import storage


def upload_run(run_dir):
    """
    Upload a run directory to Google Cloud Storage.

    Requirements
    ------------
    - Environment variable GCS_BUCKET must be defined
    - User must be authenticated to GCP
    """

    bucket_name = os.getenv("GCS_BUCKET")

    if not bucket_name:
        raise ValueError(
            "La variable d'environnement GCS_BUCKET n'est pas définie.\n"
            "Exemple : export GCS_BUCKET=powerforecast-runs"
        )

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(run_dir):

        for file in files:

            local_path = os.path.join(root, file)

            blob_path = f"runs/{os.path.basename(run_dir)}/{file}"

            blob = bucket.blob(blob_path)

            blob.upload_from_filename(local_path)

            print(f"⬆️ Upload : gs://{bucket_name}/{blob_path}")

    print("\n✅ Upload terminé vers Google Cloud Storage.\n")