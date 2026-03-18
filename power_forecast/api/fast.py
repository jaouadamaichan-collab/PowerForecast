from pathlib import Path
import pickle
from power_forecast.params import *
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from power_forecast.logic.models.registry import load_model_ml, load_df
from power_forecast.logic.utils.upload_run import upload_run
from tensorflow.keras.models import load_model as load_keras_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config modèles ─────────────────────────────────────────────────────────
MODELS_DIR = Path("power_forecast/donnees/saved_models")

XGB_MODEL_NAME    = "model_xgb_130f"
XGB_MODEL_PATH    = MODELS_DIR / XGB_MODEL_NAME
XGB_N_FEATURES    = 130

RNN_MODEL_NAME    = "model_lstm_3weeks_57f_wed.keras"
RNN_MODEL_PATH    = MODELS_DIR / RNN_MODEL_NAME
RNN_N_FEATURES    = 57

def load_model_xgb(model_path: Path = XGB_MODEL_PATH) -> object:
    if not model_path.exists():
        print(f"❌ Modèle XGB introuvable : {model_path}")
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Modèle XGB chargé : {model_path}")
    return model


try:
    app.state.model_xgb = load_model_xgb(XGB_MODEL_PATH)
except Exception as e:
    print(f"⚠️  Impossible de charger le modèle XGB : {e}")
    app.state.model_xgb = None

try:
    app.state.model_rnn = load_keras_model(RNN_MODEL_PATH)
    print(f"✅ Modèle RNN chargé : {RNN_MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Impossible de charger le modèle RNN : {e}")
    app.state.model_rnn = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_rnn_paths(objective_day: pd.Timestamp, output_length_rnn: int):
    X_new_index = pd.date_range(
        start=objective_day - pd.Timedelta(hours=INPUT_LENGTH_RNN),
        end=objective_day   - pd.Timedelta(hours=1),
        freq="h"
    )
    y_true_index = pd.date_range(
        start=objective_day,
        periods=output_length_rnn,
        freq="h"
    )
    date_start_X_str = X_new_index[0].strftime("%Y-%m-%d")
    date_end_X_str   = X_new_index[-1].strftime("%Y-%m-%d")
    date_start_y_str = y_true_index[0].strftime("%Y-%m-%d")
    date_end_y_str   = y_true_index[-1].strftime("%Y-%m-%d")

    x_new_path  = RNN_X_NEW_DATA_DIR  / f"X_new_{date_start_X_str}_{date_end_X_str}_{RNN_N_FEATURES}f_rnn.npy"
    y_true_path = RNN_y_TRUE_DATA_DIR / f"y_true_{date_start_y_str}_{date_end_y_str}_{output_length_rnn}h_rnn.npy"

    return x_new_path, y_true_path, y_true_index


def _build_xgb_paths(objective_day: pd.Timestamp, output_length: int):
    index = pd.date_range(start=objective_day, periods=output_length, freq="h")

    date_start_str = index[0].strftime("%Y-%m-%d")
    date_end_str   = index[-1].strftime("%Y-%m-%d")
    n_features     = XGB_N_FEATURES

    x_new_path  = XGB_X_NEW_DATA_DIR  / f"X_new_{date_start_str}_{date_end_str}_{n_features}f_xgb.pkl"
    y_true_path = XGB_Y_TRUE_DATA_DIR / f"y_true_{date_start_str}_{date_end_str}_{output_length}h_xgb.pkl"

    return x_new_path, y_true_path, index



def _parse_date(date: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(date)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=f"Format de date invalide : '{date}'. Utilise le format ISO 8601."
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {'status': 'connected', 'models': ['xgb', 'rnn']}


@app.get("/predict/rnn")
def predict_rnn(
    date: str = Query(..., description="Date de début ISO 8601, ex: '2024-03-20'"),
    days: int = Query(..., description="Nombre de jours à prédire, ex: 2", ge=1)
):
    """Prédit les prix via le modèle RNN uniquement."""
    if app.state.model_rnn is None:
        raise HTTPException(status_code=503, detail="Modèle RNN non chargé")

    objective_day     = _parse_date(date)
    output_length_rnn = days * 24

    x_new_path, y_true_path, y_true_index = _build_rnn_paths(objective_day, output_length_rnn)

    if not x_new_path.exists():
        raise HTTPException(status_code=404, detail=f"X_new RNN introuvable : {x_new_path}")

    X_new      = np.load(x_new_path)
    y_pred_rnn = app.state.model_rnn.predict(X_new, verbose=0).flatten().tolist()
    y_true     = np.load(y_true_path).flatten().tolist() if y_true_path.exists() else None

    predictions = [
        {
            "date":            str(ts),
            "prix_predit_rnn": round(float(p), 4),
            "prix_reel":       round(float(y), 4) if y_true else None,
        }
        for ts, p, y in zip(y_true_index, y_pred_rnn, y_true if y_true else [None] * len(y_pred_rnn))
    ]

    return {
        "date_debut":        date,
        "nb_jours":          days,
        "nb_predictions":    len(predictions),
        "unite":             "EUR/MWh",
        "y_true_disponible": y_true is not None,
        "predictions":       predictions,
    }


@app.get("/predict/xgb")
def predict_xgb(
    date: str = Query(..., description="Date de début ISO 8601, ex: '2024-03-20'"),
    days: int = Query(..., description="Nombre de jours à prédire, ex: 2", ge=1)
):
    """Prédit les prix via le modèle XGB uniquement."""
    if app.state.model_xgb is None:
        raise HTTPException(status_code=503, detail="Modèle XGB non chargé")

    objective_day = _parse_date(date)
    output_length = days * 24

    x_new_path, y_true_path, index = _build_xgb_paths(objective_day, output_length)

    if not x_new_path.exists():
        raise HTTPException(status_code=404, detail=f"X_new XGB introuvable : {x_new_path}")

    X_new      = pd.read_pickle(x_new_path)
    y_pred_xgb = app.state.model_xgb.predict(X_new).tolist()
    y_true     = pd.read_pickle(y_true_path).tolist() if y_true_path.exists() else None

    predictions = [
        {
            "date":            str(ts),
            "prix_predit_xgb": round(float(p), 4),
            "prix_reel":       round(float(y), 4) if y_true else None,
        }
        for ts, p, y in zip(index, y_pred_xgb, y_true if y_true else [None] * len(y_pred_xgb))
    ]

    return {
        "date_debut":        date,
        "nb_jours":          days,
        "nb_predictions":    len(predictions),
        "unite":             "EUR/MWh",
        "y_true_disponible": y_true is not None,
        "predictions":       predictions,
    }


@app.get("/predict/combined")
def predict_combined(
    date: str = Query(..., description="Date de début ISO 8601, ex: '2024-03-20'"),
    days: int = Query(..., description="Nombre de jours à prédire, ex: 2", ge=1)
):
    """
    Prédit les prix via RNN et XGB sur les mêmes timestamps.
    y_true : assertion que les deux sources concordent.
    Si divergence → on garde celui avec la meilleure MAE et on alerte.
    """
    if app.state.model_rnn is None:
        raise HTTPException(status_code=503, detail="Modèle RNN non chargé")
    if app.state.model_xgb is None:
        raise HTTPException(status_code=503, detail="Modèle XGB non chargé")

    objective_day = _parse_date(date)
    output_length = days * 24

    # ── RNN ───────────────────────────────────────────────────────────────
    x_new_rnn, y_true_rnn_path, y_true_index = _build_rnn_paths(objective_day, output_length)
    if not x_new_rnn.exists():
        raise HTTPException(status_code=404, detail=f"X_new RNN introuvable : {x_new_rnn}")

    X_new_rnn  = np.load(x_new_rnn)
    y_pred_rnn = app.state.model_rnn.predict(X_new_rnn, verbose=0).flatten()
    y_true_rnn = np.load(y_true_rnn_path).flatten() if y_true_rnn_path.exists() else None

    # ── XGB ───────────────────────────────────────────────────────────────
    x_new_xgb, y_true_xgb_path, _ = _build_xgb_paths(objective_day, output_length)
    if not x_new_xgb.exists():
        raise HTTPException(status_code=404, detail=f"X_new XGB introuvable : {x_new_xgb}")

    X_new_xgb  = pd.read_pickle(x_new_xgb)
    y_pred_xgb = app.state.model_xgb.predict(X_new_xgb)
    y_true_xgb = np.array(pd.read_pickle(y_true_xgb_path)) if y_true_xgb_path.exists() else None

    # ── Réconciliation y_true ─────────────────────────────────────────────
    y_true_alert = None
    y_true_final = None

    if y_true_rnn is not None and y_true_xgb is not None:
        if np.allclose(y_true_rnn, y_true_xgb, atol=1e-3):
            y_true_final = y_true_rnn
        else:
            mae_rnn = float(np.mean(np.abs(y_true_rnn - y_pred_rnn)))
            mae_xgb = float(np.mean(np.abs(y_true_xgb - y_pred_xgb)))
            y_true_final = y_true_rnn if mae_rnn <= mae_xgb else y_true_xgb
            source       = "rnn"       if mae_rnn <= mae_xgb else "xgb"
            y_true_alert = (
                f"⚠️  y_true RNN et XGB divergent — "
                f"MAE_rnn={mae_rnn:.2f}, MAE_xgb={mae_xgb:.2f} → "
                f"y_true retenu : source {source}"
            )
            print(y_true_alert)

    elif y_true_rnn is not None:
        y_true_final = y_true_rnn
    elif y_true_xgb is not None:
        y_true_final = y_true_xgb

    # ── Assemblage réponse ────────────────────────────────────────────────
    predictions = [
        {
            "date":            str(ts),
            "prix_predit_rnn": round(float(p_rnn), 4),
            "prix_predit_xgb": round(float(p_xgb), 4),
            "prix_reel":       round(float(y), 4) if y_true_final is not None else None,
        }
        for ts, p_rnn, p_xgb, y in zip(
            y_true_index,
            y_pred_rnn,
            y_pred_xgb,
            y_true_final if y_true_final is not None else [None] * output_length
        )
    ]

    return {
        "date_debut":        date,
        "nb_jours":          days,
        "nb_predictions":    len(predictions),
        "unite":             "EUR/MWh",
        "y_true_disponible": y_true_final is not None,
        "y_true_alert":      y_true_alert,
        "predictions":       predictions,
    }
