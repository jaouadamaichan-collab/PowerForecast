import os
import json
import pickle
import tempfile
from google.cloud import storage


def list_runs():
    """
    Liste tous les runs disponibles dans GCS depuis index.json.

    Returns
    -------
    runs : list of dict
        Liste des runs triés par date décroissante.

    Examples
    --------
    >>> runs = list_runs()
    """

    bucket_name = os.getenv("GCS_BUCKET")
    client      = storage.Client()
    bucket      = client.bucket(bucket_name)

    try:
        index = json.loads(bucket.blob("index.json").download_as_text())
        runs  = index.get("runs", [])
    except Exception:
        print("⚠️  index.json introuvable.")
        return []

    print(f"\n{'='*70}")
    print(f"  {'#':<4} {'run_id':<45} {'author':<12} {'test_mae':>8}")
    print(f"  {'─'*4} {'─'*45} {'─'*12} {'─'*8}")
    for i, run in enumerate(runs):
        metrics  = run.get("metrics", {})
        test_mae = metrics.get("test_mae", "—")
        test_mae = f"{test_mae:.3f}" if isinstance(test_mae, float) else test_mae
        print(f"  {i:<4} {run.get('run_id','?'):<45} "
              f"{run.get('author','?'):<12} {test_mae:>8}")
    print(f"{'='*70}\n")

    return runs


def load_run(run_id):
    """
    Télécharge et charge un modèle depuis GCS.

    Parameters
    ----------
    run_id : str
        Identifiant du run.
        Utiliser list_runs() pour voir les run_id disponibles.

    Returns
    -------
    model : objet sklearn/xgboost/keras
        Modèle chargé depuis GCS.

    Raises
    ------
    FileNotFoundError
        Si le run_id n'existe pas dans GCS.

    Examples
    --------
    >>> model = load_run("2026-03-13_13-01-12_XGBRegressor")
    """

    bucket_name = os.getenv("GCS_BUCKET")
    client      = storage.Client()
    bucket      = client.bucket(bucket_name)

    blob_path = f"runs/{run_id}/model.pkl"
    blob      = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(
            f"Run introuvable : {run_id}\n"
            f"→ Utilise list_runs() pour voir les runs disponibles."
        )

    # Téléchargement dans un fichier temporaire
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        with open(tmp.name, "rb") as f:
            model = pickle.load(f)

    print(f"✅ Modèle chargé : {run_id}")
    print(f"   Type : {type(model).__name__}")

    return model