import os
import time
import pickle

import pandas as pd
from power_forecast.params import *



"""
Fonctions utilitaires pour la sauvegarde de X_new et y_true
============================================================
Modèles supportés : RNN (format .npy) · XGB (format .pkl)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# RNN
# ──────────────────────────────────────────────────────────────────────────────

def save_rnn_data(
    X_new: np.ndarray,
    y_true: np.ndarray,
    objective_day: pd.Timestamp,
    input_length: int,
    prediction_length: int,
    base_dir: Path = Path("power_forecast/donnees"),
) -> tuple[Path, Path]:
    """
    Sauvegarde X_new et y_true du modèle RNN au format .npy.

    Nomenclature des fichiers
    -------------------------
    X_new :
        ``X_new_{date_début}_{date_fin}_{F}f_rnn.npy``

        - ``date_début``  : premier horodatage de la fenêtre d'entrée
                            (= objective_day − input_length heures)
                            format ISO  →  YYYY-MM-DD
        - ``date_fin``    : dernier horodatage de la fenêtre d'entrée
                            (= objective_day − 1 heure)
                            format ISO  →  YYYY-MM-DD
        - ``{F}f``        : nombre de features (dimension 2 du tableau 3-D),
                            le suffixe ``f`` signifie *features*
        - ``rnn``         : identifiant du modèle

        Exemple : ``X_new_2024-01-14_2024-01-15_32f_rnn.npy``
                  → fenêtre du 14 au 15 janvier 2024, 32 features

    y_true :
        ``y_true_{date_début}_{date_fin}_{H}h_rnn.npy``

        - ``date_début``  : premier horodatage de la fenêtre cible
                            (= objective_day)
                            format ISO  →  YYYY-MM-DD
        - ``date_fin``    : dernier horodatage de la fenêtre cible
                            (= objective_day + prediction_length − 1 heure)
                            format ISO  →  YYYY-MM-DD
        - ``{H}h``        : nombre d'heures prédites (horizon),
                            le suffixe ``h`` signifie *heures*
        - ``rnn``         : identifiant du modèle

        Exemple : ``y_true_2024-01-15_2024-01-15_24h_rnn.npy``
                  → 24 heures prédites le 15 janvier 2024

    Chemins de sauvegarde
    ---------------------
    - X_new  → ``{base_dir}/x_new_rnn/``
    - y_true → ``{base_dir}/y_true_rnn/``

    Parameters
    ----------
    X_new            : np.ndarray, shape (samples, input_length, features)
    y_true           : np.ndarray, shape (prediction_length,)
    objective_day    : pd.Timestamp — premier instant de la fenêtre cible
    input_length     : int — nombre d'heures dans la fenêtre d'entrée
    prediction_length: int — nombre d'heures dans la fenêtre cible
    base_dir         : Path — répertoire racine des données

    Returns
    -------
    (x_new_path, y_true_path) : chemins absolus des fichiers sauvegardés
    """
    # ── Dossiers ──────────────────────────────────────────────────────────────
    x_new_dir  = base_dir / "x_new_rnn"
    y_true_dir = base_dir / "y_true_rnn"
    x_new_dir.mkdir(parents=True, exist_ok=True)
    y_true_dir.mkdir(parents=True, exist_ok=True)

    # ── Index temporels ───────────────────────────────────────────────────────
    X_new_index = pd.date_range(
        start=objective_day - pd.Timedelta(hours=input_length),
        end=objective_day   - pd.Timedelta(hours=1),
        freq="h",
    )
    y_true_index = pd.date_range(
        start=objective_day,
        periods=prediction_length,
        freq="h",
    )

    # ── Représentations textuelles pour les noms de fichiers ──────────────────
    date_start_X_str = X_new_index[0].strftime("%Y-%m-%d")
    date_end_X_str   = X_new_index[-1].strftime("%Y-%m-%d")
    date_start_y_str = y_true_index[0].strftime("%Y-%m-%d")
    date_end_y_str   = y_true_index[-1].strftime("%Y-%m-%d")

    n_features = X_new.shape[2]   # dimension features du tableau 3-D

    # ── Chemins ───────────────────────────────────────────────────────────────
    x_new_path  = x_new_dir  / f"X_new_{date_start_X_str}_{date_end_X_str}_{n_features}f_rnn.npy"
    y_true_path = y_true_dir / f"y_true_{date_start_y_str}_{date_end_y_str}_{prediction_length}h_rnn.npy"

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    np.save(x_new_path,  X_new)
    np.save(y_true_path, y_true)

    print(f"✅ X_new  sauvegardé  → {x_new_path}")
    print(f"✅ y_true sauvegardé  → {y_true_path}")

    return x_new_path, y_true_path


# ──────────────────────────────────────────────────────────────────────────────
# XGB
# ──────────────────────────────────────────────────────────────────────────────

def save_xgb_data(
    X_new: pd.DataFrame,
    y_true: pd.Series,
    objective_day: pd.Timestamp,
    base_dir: Path = Path("power_forecast/donnees"),
) -> tuple[Path, Path]:
    """
    Sauvegarde X_new et y_true du modèle XGBoost au format .pkl.

    Nomenclature des fichiers
    -------------------------
    X_new :
        ``X_new_{date_début}_{date_fin}_{F}f_xgb.pkl``

        - ``date_début``  : premier horodatage couvert par X_new
                            (= objective_day, heure 0)
                            format ISO  →  YYYY-MM-DD
        - ``date_fin``    : dernier horodatage couvert par X_new
                            (= objective_day + len(X_new) − 1 heure)
                            format ISO  →  YYYY-MM-DD
        - ``{F}f``        : nombre de colonnes / features du tableau 2-D,
                            le suffixe ``f`` signifie *features*
        - ``xgb``         : identifiant du modèle

        Exemple : ``X_new_2024-01-15_2024-01-15_130f_xgb.pkl``
                  → 48 lignes × 130 features couvrant le 15 janvier 2024

    y_true :
        ``y_true_{date_début}_{date_fin}_{H}h_xgb.pkl``

        - ``date_début``  : premier horodatage de la cible
                            (= objective_day)
                            format ISO  →  YYYY-MM-DD
        - ``date_fin``    : dernier horodatage de la cible
                            (= objective_day + len(y_true) − 1 heure)
                            format ISO  →  YYYY-MM-DD
        - ``{H}h``        : nombre de valeurs cibles (heures prédites),
                            le suffixe ``h`` signifie *heures*
        - ``xgb``         : identifiant du modèle

        Exemple : ``y_true_2024-01-15_2024-01-15_48h_xgb.pkl``
                  → 48 heures prédites le 15 janvier 2024

    Chemins de sauvegarde
    ---------------------
    - X_new  → ``{base_dir}/x_new_xgb/``
    - y_true → ``{base_dir}/y_true_xgb/``

    Parameters
    ----------
    X_new         : np.ndarray, shape (heures, features)   — tableau 2-D
    y_true        : np.ndarray, shape (heures,)
    objective_day : pd.Timestamp — premier instant de la journée cible
    base_dir      : Path — répertoire racine des données

    Returns
    -------
    (x_new_path, y_true_path) : chemins absolus des fichiers sauvegardés
    """
    # ── Dossiers ──────────────────────────────────────────────────────────────
    x_new_dir  = base_dir / "x_new_xgb"
    y_true_dir = base_dir / "y_true_xgb"
    x_new_dir.mkdir(parents=True, exist_ok=True)
    y_true_dir.mkdir(parents=True, exist_ok=True)

    # ── Index temporels ───────────────────────────────────────────────────────
    X_new_index  = pd.date_range(start=objective_day, periods=len(X_new),  freq="h")
    y_true_index = pd.date_range(start=objective_day, periods=len(y_true), freq="h")

    # ── Représentations textuelles pour les noms de fichiers ──────────────────
    date_start_X_str = X_new_index[0].strftime("%Y-%m-%d")
    date_end_X_str   = X_new_index[-1].strftime("%Y-%m-%d")
    date_start_y_str = y_true_index[0].strftime("%Y-%m-%d")
    date_end_y_str   = y_true_index[-1].strftime("%Y-%m-%d")

    n_features = X_new.shape[1]   # dimension features du tableau 2-D
    n_hours    = len(y_true)

    # ── Chemins ───────────────────────────────────────────────────────────────
    x_new_path  = x_new_dir  / f"X_new_{date_start_X_str}_{date_end_X_str}_{n_features}f_xgb.pkl"
    y_true_path = y_true_dir / f"y_true_{date_start_y_str}_{date_end_y_str}_{n_hours}h_xgb.pkl"

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    with open(x_new_path,  "wb") as f:
        pickle.dump(X_new,  f)
    with open(y_true_path, "wb") as f:
        pickle.dump(y_true, f)

    print(f"✅ X_new  sauvegardé  → {x_new_path}")
    print(f"✅ y_true sauvegardé  → {y_true_path}")

    return x_new_path, y_true_path


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


def save_df_topickle(df):
    os.makedirs(LOCAL_REGISTRY_PATH_DF, exist_ok=True)

    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")

    filename = f"{start_date}_to_{end_date}.pkl"
    path = os.path.join(LOCAL_REGISTRY_PATH_DF, filename)

def save_df(df, name):
    os.makedirs(LOCAL_REGISTRY_PATH_DF, exist_ok=True)

    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")

    filename = f"{name}_{start_date}_to_{end_date}.pkl"
    path = os.path.join(LOCAL_REGISTRY_PATH_DF, filename)

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
