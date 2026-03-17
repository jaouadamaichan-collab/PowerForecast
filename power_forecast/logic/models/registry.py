import os
import time
import pickle
from power_forecast.params import *
from google.cloud import storage

def save_model_ml(model, model_name: str = None) -> str:
    """
    Sauvegarde le modèle localement puis l'upload sur GCS.
    Retourne le chemin GCS du blob créé.

    Le blob est stocké à : runs/{timestamp}_{model_name}/model.pkl

    Example:
        save_model_ml(model, model_name="HistGradientBoostingRegressor")
    """

    if model_name is None:
        model_name = "model"

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{model_name}"

    # Sauvegarde locale
    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    local_path = os.path.join(LOCAL_REGISTRY_PATH, f"{run_name}.pkl")
    with open(local_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved locally: {local_path}")

    # Upload sur GCS
    gcs_blob_path = f"runs/{run_name}/model.pkl"
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(local_path)
    print(f"✅ Model uploaded to GCS: gs://{GCS_BUCKET}/{gcs_blob_path}")

    return gcs_blob_path


def load_model_ml(model_name: str = None):
    """
    Charge un modèle depuis GCS.
    - Si model_name est fourni, charge le blob le plus récent dont le nom contient model_name.
    - Sinon, charge le blob le plus récent parmi tous les runs.

    Example:
        load_model_ml("HistGradientBoostingRegressor")
        load_model_ml()  # charge le plus récent
    """
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    blobs = list(bucket.list_blobs(prefix="runs/"))
    blobs = [b for b in blobs if b.name.endswith("model.pkl")]

    if not blobs:
        print(f"❌ No model found in gs://{GCS_BUCKET}/runs/")
        return None

    if model_name:
        blobs = [b for b in blobs if model_name in b.name]
        if not blobs:
            print(f"❌ No model matching '{model_name}' found in GCS")
            return None

    latest_blob = max(blobs, key=lambda b: b.updated)

    # Téléchargement local
    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    local_filename = latest_blob.name.replace("/", "_")
    local_path = os.path.join(LOCAL_REGISTRY_PATH, local_filename)
    latest_blob.download_to_filename(local_path)
    print(f"✅ Model downloaded from GCS: gs://{GCS_BUCKET}/{latest_blob.name}")

    with open(local_path, "rb") as f:
        model = pickle.load(f)

    print(f"✅ Model loaded: {latest_blob.name}")
    return model


def save_scaler(scaler, scaler_name: str = 'scaler') -> None:
    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    scaler_path = os.path.join(LOCAL_REGISTRY_PATH, f"{scaler_name}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved locally as '{scaler_name}.pkl'")


def load_scaler(scaler_name: str = 'scaler'):
    scaler_path = os.path.join(LOCAL_REGISTRY_PATH, f"{scaler_name}.pkl")
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler '{scaler_name}.pkl' not found in {LOCAL_REGISTRY_PATH}")
        return None
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"✅ Scaler loaded from {scaler_path}")
    return scaler


def save_df(df, name):
    path = Path(name)

    # Only prepend PICKLE_DIR if it's a bare filename (no directory)
    if path.parent == Path("."):
        path = Path(PICKLE_DIR) / f"{name}.pkl"
    # else: use the full path as-is, just ensure no double .pkl
    elif not path.suffix == ".pkl":
        path = path.with_suffix(".pkl")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    size = path.stat().st_size
    print(f"Saved → {path} | {size / 1024 / 1024:.2f} MB")


def load_df(name):
    """
    Load a DataFrame from a pickle file in the PICKLE_DIR folder.

    Parameters
    ----------
    name : filename without extension (e.g. 'df_enriched' → 'df_enriched.pkl')

    Returns
    -------
    df : loaded DataFrame with original index and dtypes preserved
    """
    path = f"{PICKLE_DIR}/{name}.pkl"
    df = pd.read_pickle(path)
    print(f"Loaded ← {path} | {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def save_X_test_gcs(df, name):
    from google.cloud import storage

    local_path = Path(PICKLE_DIR) / f"{name}.pkl"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(local_path)

    blob_path = f"pickle_files/{name}.pkl"
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Saved → gs://{GCS_BUCKET}/{blob_path}")


def load_X_test_gcs():
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    blobs = list(bucket.list_blobs(prefix="pickle_files/"))
    blobs = [b for b in blobs if b.name.endswith(".pkl")]

    if not blobs:
        print(f"❌ No pickle file found in gs://{GCS_BUCKET}/pickle_files/")
        return None

    latest_blob = max(blobs, key=lambda b: b.updated)

    local_path = Path(PICKLE_DIR) / Path(latest_blob.name).name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    latest_blob.download_to_filename(local_path)

    df = pd.read_pickle(local_path)
    print(f"Loaded ← gs://{GCS_BUCKET}/{latest_blob.name} | {df.shape[0]} rows x {df.shape[1]} columns")
    return df
