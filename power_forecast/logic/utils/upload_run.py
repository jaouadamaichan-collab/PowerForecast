import os
import json
from google.cloud import storage


def upload_run(run_dir):
    """
    Upload un run local vers Google Cloud Storage.
    Met à jour automatiquement index.json et runs_bq.ndjson.

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

    # --- Mise à jour runs_bq.ndjson ---
    _update_bq_export(bucket)

    print(f"\n✅ Upload terminé — gs://{bucket_name}/runs/{run_name}\n")


def _update_index(bucket, bucket_name, run_dir, run_name):
    """
    Met à jour index.json global dans GCS.
    Crée le fichier s'il n'existe pas.
    """

    index_blob = bucket.blob("index.json")

    try:
        index = json.loads(index_blob.download_as_text())
    except Exception:
        index = {"runs": []}

    run_info_path = os.path.join(run_dir, "run_info.json")
    if os.path.exists(run_info_path):
        with open(run_info_path) as f:
            run_info = json.load(f)

        index["runs"] = [
            r for r in index["runs"]
            if r.get("run_id") != run_info.get("run_id")
        ]
        index["runs"].append(run_info)
        index["runs"].sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )

    index_blob.upload_from_string(
        json.dumps(index, indent=2),
        content_type="application/json"
    )
    print(f"📋 index.json mis à jour — {len(index['runs'])} runs")


def _extract_test_mae(metrics):
    """Extrait test_mae peu importe le format des métriques."""
    if not metrics:
        return None
    if "test_mae" in metrics:
        return metrics["test_mae"]
    if "Test" in metrics:
        t = metrics["Test"]
        if isinstance(t, dict):
            return t.get("MAE") or t.get("mae")
        return t
    if "MAE" in metrics:
        m = metrics["MAE"]
        if isinstance(m, dict):
            return m.get("Test") or m.get("2")
        return m
    return None


def _update_bq_export(bucket):
    """
    Régénère runs_bq.ndjson depuis index.json.
    Utilisé par BigQuery comme source de la table runs.
    """

    try:
        data = json.loads(bucket.blob("index.json").download_as_text())
        runs = data.get("runs", [])
    except Exception:
        print("⚠️  index.json introuvable, runs_bq.ndjson non mis à jour")
        return

    ndjson_lines = []
    for run in runs:
        m    = run.get("metrics", {})
        flat = {
            "run_id"    : run.get("run_id", ""),
            "model"     : run.get("model", ""),
            "author"    : run.get("author", ""),
            "created_at": run.get("created_at", "")[:19].replace("T", " "),
            "train_mae" : m.get("train_mae"),
            "val_mae"   : m.get("val_mae"),
            "test_mae"  : _extract_test_mae(m),
            "train_rmse": m.get("train_rmse"),
            "test_rmse" : m.get("test_rmse"),
        }
        ndjson_lines.append(json.dumps(flat))

    bucket.blob("runs_bq.ndjson").upload_from_string(
        "\n".join(ndjson_lines),
        content_type="application/json"
    )
    print(f"📊 runs_bq.ndjson mis à jour — {len(runs)} lignes")