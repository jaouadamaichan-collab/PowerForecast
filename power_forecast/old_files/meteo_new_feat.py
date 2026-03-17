"""
Météo des capitales européennes — 2017 à 2025
=============================================
Sources : Open-Meteo (gratuit, sans clé API)
  - Observations réelles : ERA5 via archive-api.open-meteo.com
  - Prévisions archivées : ECMWF IFS via historical-forecast-api.open-meteo.com
    (disponibles à partir de 2021 seulement)

Installation des dépendances :
    pip install requests pandas tqdm

Utilisation :
    python fetch_european_capitals_weather.py

Le fichier CSV sera créé dans le même répertoire.
"""

import requests
import pandas as pd
import time
from tqdm import tqdm

# ── Capitales européennes ─────────────────────────────────────────────────────
CAPITALS = [{"city": "Paris", "country": "FR", "lat": 48.8566,  "lon":  2.3522}]



# ── Paramètres ─────────────────────────────────────────────────────────────────
OBS_START   = "2017-01-01"
OBS_END     = "2025-12-31"
FCST_START  = "2021-01-01"   # prévisions ECMWF disponibles à partir de 2021
FCST_END    = "2025-12-31"
LEAD_DAYS   = 3              # prévision émise 3 jours avant la date cible
MODEL       = "ecmwf_ifs025"
OUTPUT_FILE = "european_capitals_weather_2017_2025.csv"

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_mean",
]


# ── Fonctions de récupération ──────────────────────────────────────────────────
def fetch_observations(lat: float, lon: float) -> pd.DataFrame:
    """Récupère les observations ERA5 (2017–2025)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": OBS_START,
        "end_date": OBS_END,
        "daily": ",".join(DAILY_VARS),
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["daily"])
    df.rename(columns={"time": "date"}, inplace=True)
    # Renommer les colonnes avec préfixe obs_
    df.columns = ["date"] + [f"obs_{c}" for c in df.columns[1:]]
    return df


def fetch_forecast(lat: float, lon: float) -> pd.DataFrame:
    """Récupère les prévisions archivées ECMWF (2021–2025, lead=LEAD_DAYS jours)."""
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": FCST_START,
        "end_date": FCST_END,
        "daily": ",".join(DAILY_VARS),
        "models": MODEL,
        "timezone": "UTC",
        "forecast_days": 1,
        "past_days": LEAD_DAYS,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["daily"])
    df.rename(columns={"time": "date"}, inplace=True)
    # Renommer les colonnes avec préfixe fcst_
    df.columns = ["date"] + [f"fcst_{c}" for c in df.columns[1:]]
    return df


# ── Boucle principale ──────────────────────────────────────────────────────────
all_frames = []
errors = []

print(f"\nRécupération des données pour {len(CAPITALS)} capitales européennes...")
print(f"Période observations : {OBS_START} → {OBS_END}")
print(f"Période prévisions   : {FCST_START} → {FCST_END} (lead {LEAD_DAYS}j, modèle {MODEL})\n")

for cap in tqdm(CAPITALS, desc="Capitales", unit="ville"):
    city    = cap["city"]
    country = cap["country"]
    lat     = cap["lat"]
    lon     = cap["lon"]

    try:
        df_obs = fetch_observations(lat, lon)
        time.sleep(0.4)   # pause pour respecter le rate limit de l'API
    except Exception as e:
        errors.append(f"{city}: observations — {e}")
        df_obs = None

    try:
        df_fcst = fetch_forecast(lat, lon)
        time.sleep(0.4)
    except Exception as e:
        errors.append(f"{city}: prévisions — {e}")
        df_fcst = None

    if df_obs is None:
        continue  # impossible de construire la ligne sans observations

    if df_fcst is not None:
        df = pd.merge(df_obs, df_fcst, on="date", how="left")
    else:
        df = df_obs.copy()
        for v in DAILY_VARS:
            df[f"fcst_{v}"] = None

    df.insert(0, "country", country)
    df.insert(0, "city", city)
    df.insert(2, "lat", lat)
    df.insert(3, "lon", lon)
    all_frames.append(df)


# ── Assemblage et export ───────────────────────────────────────────────────────
if not all_frames:
    print("\nAucune donnée récupérée — vérifiez votre connexion Internet.")
else:
    final = pd.concat(all_frames, ignore_index=True)

    # Ordre des colonnes
    ordered_cols = [
        "city", "country", "lat", "lon", "date",
        "obs_temperature_2m_mean", "obs_temperature_2m_max", "obs_temperature_2m_min",
        "obs_precipitation_sum", "obs_windspeed_10m_max", "obs_relative_humidity_2m_mean",
        "fcst_temperature_2m_mean", "fcst_temperature_2m_max", "fcst_temperature_2m_min",
        "fcst_precipitation_sum", "fcst_windspeed_10m_max", "fcst_relative_humidity_2m_mean",
    ]
    final = final[[c for c in ordered_cols if c in final.columns]]

    # Calcul des métriques d'erreur (uniquement sur 2021–2025)
    for var in ["temperature_2m_mean", "precipitation_sum", "windspeed_10m_max"]:
        obs_col  = f"obs_{var}"
        fcst_col = f"fcst_{var}"
        if obs_col in final.columns and fcst_col in final.columns:
            final[f"err_{var}"]     = final[fcst_col] - final[obs_col]
            final[f"abs_err_{var}"] = final[f"err_{var}"].abs()

    final.to_csv(OUTPUT_FILE, index=False, float_format="%.2f")

    print(f"\n✓ Fichier créé : {OUTPUT_FILE}")
    print(f"  Dimensions   : {final.shape[0]:,} lignes × {final.shape[1]} colonnes")
    print(f"  Villes       : {final['city'].nunique()}")
    print(f"  Période      : {final['date'].min()} → {final['date'].max()}")
    print(f"\nAperçu (5 premières lignes) :")
    print(final.head().to_string(index=False))

if errors:
    print(f"\n⚠ Erreurs rencontrées ({len(errors)}) :")
    for e in errors:
        print(f"  - {e}")
