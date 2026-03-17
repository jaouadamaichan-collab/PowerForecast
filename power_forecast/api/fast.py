import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
# from power_forecast.logic.preprocessing.preprocessor import preproc_histxgb_X_new
from power_forecast.logic.models.registry import load_model_ml, load_df
from power_forecast.logic.utils.upload_run import upload_run

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement au démarrage ────────────────────────────────────────────────────
app.state.model = load_model_ml()

try:
    app.state.df_cache = load_df('X_test')
    print("✅ Feature DataFrame chargé depuis le cache pickle")
except Exception as e:
    print(f"⚠️  Impossible de charger le cache pickle : {e}")
    app.state.df_cache = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {'status': 'connected', 'model': 'top_model'}


@app.post("/predict")
def predict(data: dict):
    """
    Prédit le prix à partir d'un X_new déjà feature-engineered.
    Le dict doit contenir toutes les colonnes utilisées à l'entraînement.
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    X_new = pd.DataFrame([data])
    y_pred = app.state.model.predict(X_new)

    return {
        'prix_predit': float(y_pred[0]),
        'unite': 'EUR/MWh'
    }


@app.get("/predict/from_cache")
def predict_from_cache(
    date: str = Query(..., description="Date de début ISO 8601, ex: '2024-01-01'"),
    days: int = Query(..., description="Nombre de jours à prédire, ex: 2", ge=1)
):
    """
    Prédit les prix heure par heure sur une période donnée.
    Exemple : date='2024-01-01', days=2 → 48 prédictions horaires.
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if app.state.df_cache is None:
        raise HTTPException(
            status_code=503,
            detail="Cache non disponible — lance d'abord build_feature_dataframe()"
        )

    try:
        date_parsed = pd.Timestamp(date)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=f"Format de date invalide : '{date}'. Utilise le format ISO 8601."
        )

    df = app.state.df_cache

    # Normalise les timezones si besoin
    if df.index.tz is None and date_parsed.tz is not None:
        date_parsed = date_parsed.tz_localize(None)
    elif df.index.tz is not None and date_parsed.tz is None:
        date_parsed = date_parsed.tz_localize(df.index.tz)

    # Génère les timestamps horaires sur la période demandée
    timestamps = pd.date_range(start=date_parsed, periods=days * 24, freq='h')

    # Vérifie que toutes les heures sont dans le cache
    missing = [str(t) for t in timestamps if t not in df.index]
    if missing:
        available = df.index[[0, -1]]
        raise HTTPException(
            status_code=404,
            detail=(
                f"{len(missing)} heure(s) introuvable(s) dans le cache "
                f"(ex: {missing[0]}). "
                f"Plage disponible : {available[0]} → {available[-1]}"
            )
        )

    # Extrait toutes les lignes d'un coup et prédit en batch
    X_new = df.loc[timestamps]
    y_pred = app.state.model.predict(X_new)

    predictions = [
        {'date': str(ts), 'prix_predit': float(price)}
        for ts, price in zip(timestamps, y_pred)
    ]

    return {
        'date_debut': str(date_parsed),
        'nb_jours': days,
        'nb_predictions': len(predictions),
        'unite': 'EUR/MWh',
        'predictions': predictions
    }
