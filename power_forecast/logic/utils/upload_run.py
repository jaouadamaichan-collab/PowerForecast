import os
import json
from google.cloud import storage


def upload_run(run_dir):
    """
    Upload un run local vers Google Cloud Storage.
    Met à jour automatiquement index.json global.

    Parameters
    ----------
    run_dir : str
        Chemin local du run (retourné par save_run).

    Examples
    --------
    >>> upload_run(run_path)
    """

    bucket_name = os.getenv("GCS_BUCKET")
    if not bucket_name:
        raise ValueError(
            "Variable d'environnement GCS_BUCKET non définie.\n"
            "Exemple : os.environ['GCS_BUCKET'] = 'powerforecast-ml-runs'"
        )

    client   = storage.Client()
    bucket   = client.bucket(bucket_name)
    run_name = os.path.basename(run_dir)

    # --- Upload des fichiers du run ---
    for root, _, files in os.walk(run_dir):
        for file in files:
            local_path = os.path.join(root, file)
            blob_path  = f"runs/{run_name}/{file}"
            bucket.blob(blob_path).upload_from_filename(local_path)
            print(f"⬆️  gs://{bucket_name}/{blob_path}")

    # --- Mise à jour index.json ---
    _update_index(bucket, bucket_name, run_dir, run_name)

    print(f"\n✅ Upload terminé — gs://{bucket_name}/runs/{run_name}\n")


def _update_index(bucket, bucket_name, run_dir, run_name):
    """
    Met à jour index.json global dans GCS.
    Crée le fichier s'il n'existe pas.
    """

    index_blob = bucket.blob("index.json")

    # Charger l'index existant ou créer un nouveau
    try:
        index = json.loads(index_blob.download_as_text())
    except Exception:
        index = {"runs": []}

    # Lire run_info.json local
    run_info_path = os.path.join(run_dir, "run_info.json")
    if os.path.exists(run_info_path):
        with open(run_info_path) as f:
            run_info = json.load(f)

        # Éviter les doublons
        index["runs"] = [
            r for r in index["runs"]
            if r.get("run_id") != run_info.get("run_id")
        ]
        index["runs"].append(run_info)

        # Trier par date décroissante
        index["runs"].sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )

    index_blob.upload_from_string(
        json.dumps(index, indent=2),
        content_type="application/json"
    )
    print(f"📋 index.json mis à jour — {len(index['runs'])} runs")