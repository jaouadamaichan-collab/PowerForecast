import os
import time
import pickle

import pandas as pd
from power_forecast.params import *

def save_model_ml(model, model_name: str = None) -> str:
    """
    Sauvegarde le modèle localement.
    Retourne le chemin local du fichier créé.

    Example:
        save_model_ml(model, model_name="HistGradientBoostingRegressor")
    """

    if model_name is None:
        model_name = "model"

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{model_name}"

    os.makedirs(LOCAL_REGISTRY_PATH_MODELS, exist_ok=True)
    local_path = os.path.join(LOCAL_REGISTRY_PATH_MODELS, f"{run_name}.pkl")
    with open(local_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved locally: {local_path}")

    return local_path


def load_model_ml(model_name: str = None):
    """
    Charge le modèle local le plus récent.
    - Si model_name est fourni, charge le fichier le plus récent dont le nom contient model_name.
    - Sinon, charge le fichier le plus récent parmi tous les runs.

    Example:
        load_model_ml("HistGradientBoostingRegressor")
        load_model_ml()  # charge le plus récent
    """
    if not os.path.exists(LOCAL_REGISTRY_PATH_MODELS):
        print(f"❌ No model found in {LOCAL_REGISTRY_PATH_MODELS}")
        return None

    files = [f for f in os.listdir(LOCAL_REGISTRY_PATH_MODELS) if f.endswith(".pkl")]

    if not files:
        print(f"❌ No model found in {LOCAL_REGISTRY_PATH_MODELS}")
        return None

    if model_name:
        files = [f for f in files if model_name in f]
        if not files:
            print(f"❌ No model matching '{model_name}' found in {LOCAL_REGISTRY_PATH_MODELS}")
            return None

    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(LOCAL_REGISTRY_PATH_MODELS, f)))
    local_path = os.path.join(LOCAL_REGISTRY_PATH_MODELS, latest_file)

    with open(local_path, "rb") as f:
        model = pickle.load(f)

    print(f"✅ Model loaded: {local_path}")
    return model


# def save_scaler(scaler, scaler_name: str = 'scaler') -> None:
#     os.makedirs(LOCAL_REGISTRY_PATH_SC, exist_ok=True)
#     scaler_path = os.path.join(LOCAL_REGISTRY_PATH_SC, f"{scaler_name}.pkl")
#     with open(scaler_path, "wb") as f:
#         pickle.dump(scaler, f)
#     print(f"✅ Scaler saved locally as '{scaler_name}.pkl'")


# def load_scaler(scaler_name: str = 'scaler'):
#     scaler_path = os.path.join(LOCAL_REGISTRY_PATH_SC, f"{scaler_name}.pkl")
#     if not os.path.exists(scaler_path):
#         print(f"❌ Scaler '{scaler_name}.pkl' not found in {LOCAL_REGISTRY_PATH_SC}")
#         return None
#     with open(scaler_path, "rb") as f:
#         scaler = pickle.load(f)
#     print(f"✅ Scaler loaded from {scaler_path}")
#     return scaler


def save_df(df, name):
    os.makedirs(LOCAL_REGISTRY_PATH_DF, exist_ok=True)
    path = os.path.join(LOCAL_REGISTRY_PATH_DF, f"{name}.pkl")
    df.to_pickle(path)
    print(f"✅ DataFrame saved locally: {path}")


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
    path = os.path.join(LOCAL_REGISTRY_PATH_DF, f"{name}.pkl")
    df = pd.read_pickle(path)
    print(f"Loaded ← {path} | {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# def save_X_test_gcs(df, name):
#     from google.cloud import storage

#     local_path = Path(PICKLE_DIR) / f"{name}.pkl"
#     local_path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_pickle(local_path)

#     blob_path = f"pickle_files/{name}.pkl"
#     client = storage.Client()
#     bucket = client.bucket(GCS_BUCKET)
#     blob = bucket.blob(blob_path)
#     blob.upload_from_filename(local_path)
#     print(f"Saved → gs://{GCS_BUCKET}/{blob_path}")


# def load_X_test_gcs():
#     client = storage.Client()
#     bucket = client.bucket(GCS_BUCKET)

#     blobs = list(bucket.list_blobs(prefix="pickle_files/"))
#     blobs = [b for b in blobs if b.name.endswith(".pkl")]

#     if not blobs:
#         print(f"❌ No pickle file found in gs://{GCS_BUCKET}/pickle_files/")
#         return None

#     latest_blob = max(blobs, key=lambda b: b.updated)

#     local_path = Path(PICKLE_DIR) / Path(latest_blob.name).name
#     local_path.parent.mkdir(parents=True, exist_ok=True)
#     latest_blob.download_to_filename(local_path)

#     df = pd.read_pickle(local_path)
#     print(f"Loaded ← gs://{GCS_BUCKET}/{latest_blob.name} | {df.shape[0]} rows x {df.shape[1]} columns")
#     return df
