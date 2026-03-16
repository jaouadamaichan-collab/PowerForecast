import urllib.request
import urllib.parse
import json
import pandas as pd
from power_forecast.params import *
import time



def geocode_city(city_name: str) -> dict:
    """Recherche les coordonnées et le pays d'une ville via l'API Open-Meteo Geocoding."""
    url = (
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={urllib.parse.quote(city_name)}&count=1&language=fr&format=json"
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())

    if not data.get("results"):
        raise ValueError(f"Ville introuvable : '{city_name}'")

    result = data["results"][0]
    return {
        "name":    result["name"],
        "country": result.get("country_code", "??"),
        "lat":     result["latitude"],
        "lon":     result["longitude"],
    }


def fetch_historical(city: dict, date_start: str, date_end: str) -> pd.DataFrame:
    """Récupère les données historiques et retourne un DataFrame pandas."""
    hourly_variables = [
        "temperature_2m", "precipitation", "windspeed_10m", "windgusts_10m",
        "winddirection_10m", "shortwave_radiation", "weathercode",
    ]
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={city['lat']}&longitude={city['lon']}"
        f"&start_date={date_start}&end_date={date_end}"
        f"&hourly={','.join(hourly_variables)}"
        f"&timezone=UTC"
    )

    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())

    hourly_data = data["hourly"]

    df = pd.DataFrame({
        "date":                   pd.to_datetime(hourly_data["time"]),
        "ville":                  city["name"],
        "pays":                   city["country"],
        "temperature_c":       hourly_data["temperature_2m"],
        "precipitation_mm":    hourly_data["precipitation"],
        "vent_km_h":            hourly_data["windspeed_10m"],
        "rafales_km_h":         hourly_data["windgusts_10m"],
        "irradiation_MJ_m2": hourly_data["shortwave_radiation"],
        "code_meteo":       hourly_data["weathercode"],
    })

    df["Conditions"] = df["code_meteo"].map(WMO_LABELS).fillna("Inconnu")

    return df


def build_dataframe(city_names: list, date_debut: str, date_fin: str) -> pd.DataFrame:
    """Géocode les villes, récupère les données et retourne un DataFrame agrégé."""
    frames = []
    for name in city_names:
        try:
            city = geocode_city(name)
            df = fetch_historical(city, date_debut, date_fin)
            frames.append(df)
            print(f"  ✓ {city['name']} ({city['country']}) — {len(df) // 24} jours (soit {len(df)} heures)")
        except Exception as e:
            print(f"  ✘ {name} — Erreur : {e}")
        time.sleep(1)

    if not frames:
        raise RuntimeError("Aucune donnée récupérée.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["code_meteo"] = df_all["code_meteo"].astype("Int64")
    df_all = df_all.sort_values(["date", "ville"]).reset_index(drop=True)

    return df_all


def preproc_meteo(
    df: pd.DataFrame,
    date_start: str,
    date_end: str,
    city: str | list,
) -> pd.DataFrame:
    """
    Construit un DataFrame pivotté avec une ligne par date et des colonnes
    nommées  <code_ISO>_<indicateur>  pour chaque ville demandée.

    Paramètres
    ----------
    df         : DataFrame produit par build_dataframe().
    date_start : Date de début au format 'YYYY-MM-DD' (incluse).
    date_end   : Date de fin   au format 'YYYY-MM-DD' (incluse).
    city       : Ville(s) parmi VILLES_DISPONIBLES, ou "all".
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if isinstance(city, str):
        cities = VILLES_DISPONIBLES if city.lower() == "all" else [city]
    else:
        cities = list(city)

    invalides = [c for c in cities if c not in VILLES_DISPONIBLES]
    if invalides:
        raise ValueError(
            f"Ville(s) inconnue(s) : {invalides}. "
            f"Choisir parmi : {VILLES_DISPONIBLES}"
        )

    mask = (
        (df["date"] >= pd.to_datetime(date_start))
        & (df["date"] <= pd.to_datetime(date_end))
        & (df["ville"].isin(cities))
    )
    df_filtered = df.loc[mask, ["date", "ville"] + COLONNES_METEO].copy()

    if df_filtered.empty:
        raise ValueError(
            f"Aucune donnée trouvée pour la période {date_start} → {date_end} "
            f"et les villes {cities}."
        )

    df_filtered["ville"] = df_filtered["ville"].map(VILLE_TO_ISO)
    cities_iso = [VILLE_TO_ISO[c] for c in cities]

    df_pivot = df_filtered.pivot(
        index="date",
        columns="ville",
        values=COLONNES_METEO,
    )
    df_pivot.columns = [f"{iso}_{indicateur}" for indicateur, iso in df_pivot.columns]

    cols_ordonnes = [
        f"{iso}_{ind}"
        for iso in cities_iso
        for ind in COLONNES_METEO
    ]
    df_pivot = df_pivot[cols_ordonnes]
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={"date": "timestamp"}, inplace=True)
    df_pivot.sort_values("timestamp", inplace=True)
    df_pivot.reset_index(drop=True, inplace=True)
    df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"], utc=True)

    return df_pivot


def get_meteo(
    country: str | list,
    date_start: str,
    date_end: str,
    time_interval: str = "H",
) -> pd.DataFrame:

    cities = VILLES_DISPONIBLES if (isinstance(country, str) and country.lower() == "all") else (
        [country] if isinstance(country, str) else list(country)
    )

    invalides = [c for c in cities if c not in VILLES_DISPONIBLES]
    if invalides:
        raise ValueError(f"Ville(s) inconnue(s) : {invalides}. Choisir parmi : {VILLES_DISPONIBLES}")

    df_raw   = build_dataframe(cities, date_start, date_end)
    df_pivot = preproc_meteo(df_raw, date_start, date_end, country)
    df_pivot = df_pivot.set_index("timestamp")

    if time_interval == "D":
        df_pivot = df_pivot.resample("D").mean()

    df_pivot = df_pivot[~df_pivot.index.duplicated(keep='first')]
    print(f"  ✓ Meteo : {df_pivot.shape}")
    return df_pivot