# import urllib.request
# import json
# import pandas as pd
# from datetime import datetime
# import time # Import the time module

# # ── Villes européennes (ajouter/supprimer selon besoin) ──────────────────────
# CITIES = [
#     {"name": "Berlin",     "country": "Allemagne",   "lat": 52.5200,  "lon": 13.4050},
# ]

# # "name": "Paris",      "country": "France",      "lat": 48.8566,  "lon":  2.3522
# # "name": "Berlin",     "country": "Allemagne",   "lat": 52.5200,  "lon": 13.4050
# # "name": "Madrid",     "country": "Espagne",     "lat": 40.4168,  "lon": -3.7038
# # "name": "Rome",       "country": "Italie",      "lat": 41.9028,  "lon": 12.4964
# # "name": "Amsterdam",  "country": "Pays-Bas",    "lat": 52.3676,  "lon":  4.9041
# # "name": "Bruxelles",  "country": "Belgique",    "lat": 50.8503,  "lon":  4.3517
# # "name": "Vienne",     "country": "Autriche",    "lat": 48.2082,  "lon": 16.3738
# # "name": "Lisbonne",   "country": "Portugal",    "lat": 38.7223,  "lon": -9.1393
# # "name": "Stockholm",  "country": "Suède",       "lat": 59.3293,  "lon": 18.0686
# # "name": "Varsovie",   "country": "Pologne",     "lat": 52.2297,  "lon": 21.0122

# # ── Sélection active : changer ici pour basculer entre les deux listes ────────
# CITIES_ACTIVE = CITIES                    # ← capitales européennes


# # ── Plage de dates (format YYYY-MM-DD) ───────────────────────────────────────
# DATE_DEBUT = "2015-01-01"   # ← modifier ici
# DATE_FIN   = "2024-12-31"   # ← modifier ici

# # ── Fichier de sortie ─────────────────────────────────────────────────────────
# # OUTPUT_FILE = "meteo_historique_europe.csv"

# # ── Correspondance codes WMO ──────────────────────────────────────────────────
# WMO_LABELS = {
#     0: "Ciel dégagé", 1: "Principalement dégagé", 2: "Partiellement nuageux",
#     3: "Couvert", 45: "Brouillard", 48: "Brouillard givrant",
#     51: "Bruine légère", 53: "Bruine modérée", 55: "Bruine dense",
#     61: "Pluie légère", 63: "Pluie modérée", 65: "Pluie forte",
#     71: "Neige légère", 73: "Neige modérée", 75: "Neige forte",
#     80: "Averses légères", 81: "Averses modérées", 82: "Averses violentes",
#     95: "Orage", 96: "Orage avec grêle", 99: "Orage avec forte grêle",
# }


# def fetch_historical(city: dict, date_start: str, date_end: str) -> pd.DataFrame:
#     """Récupère les données historiques et retourne un DataFrame pandas."""
#     # Request hourly instantaneous variables
#     hourly_variables = [
#         "temperature_2m", "precipitation", "windspeed_10m", "windgusts_10m",
#         "winddirection_10m", "shortwave_radiation", "weathercode",
#     ]
#     url = (
#         f"https://archive-api.open-meteo.com/v1/archive"
#         f"?latitude={city['lat']}&longitude={city['lon']}"
#         f"&start_date={date_start}&end_date={date_end}"
#         f"&hourly={','.join(hourly_variables)}"
#         f"&timezone=Europe%2FParis"
#     )

#     with urllib.request.urlopen(url, timeout=30) as resp:
#         data = json.loads(resp.read())

#     hourly_data = data["hourly"]

#     # Construire le DataFrame directement depuis le JSON avec des noms de colonnes adaptés à l'heure
#     df = pd.DataFrame({
#         "Date":                   pd.to_datetime(hourly_data["time"]),
#         # "Ville":                  city["name"],
#         "Pays":                   city["country"],
#         # "Région":                 city.get("region", ""),
#         # "Latitude":               city["lat"],
#         # "Longitude":              city["lon"],
#         "Température (°C)":       hourly_data["temperature_2m"],
#         "Précipitations (mm)":    hourly_data["precipitation"],
#         "Vent (km/h)":            hourly_data["windspeed_10m"],
#         "Rafales (km/h)":         hourly_data["windgusts_10m"],
#         # "Direction_vent ()":     hourly_data["winddirection_10m"],
#         "Ensoleillement (MJ/m)": hourly_data["shortwave_radiation"],
#         # "Code météo (WMO)":       hourly_data["weathercode"],
#     })

#     # Colonne conditions lisibles
#     df["Conditions"] = df["Code météo (WMO)"].map(WMO_LABELS).fillna("Inconnu")

#     return df


# def build_dataframe() -> pd.DataFrame:
#     """Agrège les données de toutes les villes en un seul DataFrame."""
#     frames = []
#     for city in CITIES_ACTIVE:
#         try:
#             df = fetch_historical(city, DATE_DEBUT, DATE_FIN)
#             frames.append(df)
#             # Adjust message to reflect hourly data fetched (24 times days)
#             print(f"  ✓ {city['name']} ({city['country']}) — {len(df) // 24} jours (soit {len(df)} heures)")
#         except Exception as e:
#             print(f"  ✘ {city['name']} — Erreur : {e}")
#         time.sleep(10) # Add a 1-second delay between API calls

#     if not frames:
#         raise RuntimeError("Aucune donnée récupérée.")

#     # Concaténation + typage
#     df_all = pd.concat(frames, ignore_index=True)
#     df_all["Code météo (WMO)"] = df_all["Code météo (WMO)"].astype("Int64")

#     # Tri par date et ville
#     df_all = df_all.sort_values(["Date", "Ville"]).reset_index(drop=True)

#     return df_all


# def apercu(df: pd.DataFrame) -> None:
#     """Affiche un aperçu et des statistiques du DataFrame."""
#     print("\n── Aperçu (5 premières lignes) ─────────────────────────────────────")
#     print(df.head().to_string(index=False))

#     print("\n── Types des colonnes ───────────────────────────────────────────────")
#     print(df.dtypes.to_string())

#     print("\n── Statistiques descriptives ────────────────────────────────────────")
#     # Update num_cols to reflect new hourly column names
#     num_cols = ["Température (°C)",
#                 "Précipitations (mm)", "Vent (km/h)", "Ensoleillement (MJ/m)"]
#     print(df[num_cols].describe().round(2).to_string())

#     print("\n── Températures moyennes par pays ───────────────────────────────────")
#     print(
#         df.groupby("Pays")["Température (°C)"] # Use new temperature column
#         .mean().round(2)
#         .sort_values(ascending=False)
#         .to_string()
#     )


#     # ── Construction du DataFrame ─────────────────────────────────────────────
#     df = build_dataframe()

# # Chargement du dataset source
# RAW_PATH = "meteo_historique_europe_v1.csv"

# # Mapping des colonnes source → colonnes cibles (suffixe ville)
# COLONNES_METEO = {
#     "Temp_moy (°C)":        "Température (°C)",
#     "Précipitations (mm)":  "Précipitations (mm)",
#     "Vent_max (km/h)":      "Vent (km/h)",
#     "Rafales_max (km/h)":   "Rafales (km/h)",
#     "Ensoleillement (MJ/m)":"Ensoleillement (MJ/m)",
# }

# VILLES_DISPONIBLES = ["Berlin", "Bruxelles", "Lisbonne", "Madrid", "Paris", "Vienne"]


# def preproc_meteo(
#     date_start: str,
#     date_end: str,
#     city: str | list[str],
#     path: str = RAW_PATH,
# ) -> pd.DataFrame:
#     """
#     Construit un DataFrame pivotté avec une ligne par date et des colonnes
#     nommées  <ville>_<indicateur>  pour chaque ville demandée.

#     Paramètres
#     ----------
#     date_start : str
#         Date de début au format 'YYYY-MM-DD' (incluse).
#     date_end : str
#         Date de fin   au format 'YYYY-MM-DD' (incluse).
#     city : str ou liste de str
#         Ville(s) parmi : Berlin, Bruxelles, Lisbonne, Madrid, Paris, Vienne.
#         Passer "all" pour toutes les villes.
#     path : str
#         Chemin vers le fichier CSV source.

#     Retourne
#     --------
#     pd.DataFrame
#         Colonnes : timestamp | <ville>_Température (°C) |
#                    <ville>_Précipitations (mm) | <ville>_Vent (km/h) |
#                    <ville>_Rafales (km/h) | <ville>_Ensoleillement (MJ/m)
#     """
#     # ── 1. Chargement ────────────────────────────────────────────────────────
#     df["Date"] = pd.to_datetime(df["Date"])

#     # ── 2. Normalisation des paramètres d'entrée ─────────────────────────────
#     if isinstance(city, str):
#         cities = VILLES_DISPONIBLES if city.lower() == "all" else [city]
#     else:
#         cities = list(city)

#     # Validation des villes
#     invalides = [c for c in cities if c not in VILLES_DISPONIBLES]
#     if invalides:
#         raise ValueError(
#             f"Ville(s) inconnue(s) : {invalides}. "
#             f"Choisir parmi : {VILLES_DISPONIBLES}"
#         )

#     # ── 3. Filtrage temporel ──────────────────────────────────────────────────
#     mask = (
#         (df["Date"] >= pd.to_datetime(date_start))
#         & (df["Date"] <= pd.to_datetime(date_end))
#         & (df["Ville"].isin(cities))
#     )
#     df_filtered = df.loc[mask, ["Date", "Ville"] + list(COLONNES_METEO.keys())].copy()

#     if df_filtered.empty:
#         raise ValueError(
#             f"Aucune donnée trouvée pour la période {date_start} → {date_end} "
#             f"et les villes {cities}."
#         )

#     # ── 4. Renommage des colonnes métriques ───────────────────────────────────
#     df_filtered.rename(columns=COLONNES_METEO, inplace=True)

#     # ── 5. Pivot : une ligne par date, une colonne par (ville × indicateur) ───
#     df_pivot = df_filtered.pivot(
#         index="Date",
#         columns="Ville",
#         values=list(COLONNES_METEO.values()),
#     )

#     # Aplatir le MultiIndex : "Température (°C)_Paris" → "Paris_Température (°C)"
#     df_pivot.columns = [f"{ville}_{indicateur}" for indicateur, ville in df_pivot.columns]

#     # Trier les colonnes : regroupées par ville dans l'ordre de COLONNES_METEO
#     cols_ordonnes = [
#         f"{ville}_{ind}"
#         for ville in cities
#         for ind in COLONNES_METEO.values()
#     ]
#     df_pivot = df_pivot[cols_ordonnes]

#     # ── 6. Remise en forme finale ─────────────────────────────────────────────
#     df_pivot.reset_index(inplace=True)
#     df_pivot.rename(columns={"Date": "timestamp"}, inplace=True)
#     df_pivot.sort_values("timestamp", inplace=True)
#     df_pivot.reset_index(drop=True, inplace=True)

#     return df_pivot


# # ── Exemple d'utilisation ─────────────────────────────────────────────────────
# if __name__ == "__main__":
#     # Une seule ville
#     df_paris = preproc_meteo("2022-06-01", "2022-06-30", "Paris")
#     print("=== Paris – juin 2022 ===")
#     print(df_paris.head())

#     # Plusieurs villes
#     df_multi = preproc_meteo("2023-01-01", "2023-01-10", ["Paris", "Madrid", "Berlin"])
#     print("\n=== Paris / Madrid / Berlin – 10 premiers jours 2023 ===")
#     print(df_multi.to_string())

#     # Toutes les villes
#     df_all = preproc_meteo("2024-07-01", "2024-07-03", "all")
#     print("\n=== Toutes les villes – 1-3 juillet 2024 ===")
#     print(df_all.to_string())


import urllib.request
import json
import pandas as pd
from datetime import datetime
import time

# ── Villes européennes (ajouter/supprimer selon besoin) ──────────────────────
# Codes pays : ISO 3166-1 alpha-2
CITIES = [
    {"name": "Berlin",     "country": "DE",  "lat": 52.5200,  "lon": 13.4050},
    # {"name": "Paris",      "country": "FR",  "lat": 48.8566,  "lon":  2.3522},
    # {"name": "Madrid",     "country": "ES",  "lat": 40.4168,  "lon": -3.7038},
    # {"name": "Rome",       "country": "IT",  "lat": 41.9028,  "lon": 12.4964},
    # {"name": "Amsterdam",  "country": "NL",  "lat": 52.3676,  "lon":  4.9041},
    # {"name": "Bruxelles",  "country": "BE",  "lat": 50.8503,  "lon":  4.3517},
    # {"name": "Vienne",     "country": "AT",  "lat": 48.2082,  "lon": 16.3738},
    # {"name": "Lisbonne",   "country": "PT",  "lat": 38.7223,  "lon": -9.1393},
    # {"name": "Stockholm",  "country": "SE",  "lat": 59.3293,  "lon": 18.0686},
    # {"name": "Varsovie",   "country": "PL",  "lat": 52.2297,  "lon": 21.0122},
]

CITIES_ACTIVE = CITIES

# ── Plage de dates (format YYYY-MM-DD) ───────────────────────────────────────
DATE_DEBUT = "2024-01-01"
DATE_FIN   = "2024-12-31"

# ── Correspondance codes WMO ──────────────────────────────────────────────────
WMO_LABELS = {
    0: "Ciel dégagé", 1: "Principalement dégagé", 2: "Partiellement nuageux",
    3: "Couvert", 45: "Brouillard", 48: "Brouillard givrant",
    51: "Bruine légère", 53: "Bruine modérée", 55: "Bruine dense",
    61: "Pluie légère", 63: "Pluie modérée", 65: "Pluie forte",
    71: "Neige légère", 73: "Neige modérée", 75: "Neige forte",
    80: "Averses légères", 81: "Averses modérées", 82: "Averses violentes",
    95: "Orage", 96: "Orage avec grêle", 99: "Orage avec forte grêle",
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
        f"&timezone=Europe%2FParis"
    )

    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())

    hourly_data = data["hourly"]

    # FIX 1 : "Code météo (WMO)" était référencé mais jamais créé dans le DataFrame
    df = pd.DataFrame({
        "Date":                   pd.to_datetime(hourly_data["time"]),
        "Ville":                  city["name"],        # FIX 2 : colonne "Ville" commentée par erreur (nécessaire pour le tri et le pivot)
        "Pays":                   city["country"],
        "Température (°C)":       hourly_data["temperature_2m"],
        "Précipitations (mm)":    hourly_data["precipitation"],
        "Vent (km/h)":            hourly_data["windspeed_10m"],
        "Rafales (km/h)":         hourly_data["windgusts_10m"],
        "Ensoleillement (MJ/m)":  hourly_data["shortwave_radiation"],
        "Code météo (WMO)":       hourly_data["weathercode"],  # FIX 1 suite : ajout de la clé manquante
    })

    df["Conditions"] = df["Code météo (WMO)"].map(WMO_LABELS).fillna("Inconnu")

    return df


def build_dataframe() -> pd.DataFrame:
    """Agrège les données de toutes les villes en un seul DataFrame."""
    frames = []
    for city in CITIES_ACTIVE:
        try:
            df = fetch_historical(city, DATE_DEBUT, DATE_FIN)
            frames.append(df)
            print(f"  ✓ {city['name']} ({city['country']}) — {len(df) // 24} jours (soit {len(df)} heures)")
        except Exception as e:
            print(f"  ✘ {city['name']} — Erreur : {e}")
        time.sleep(10)

    if not frames:
        raise RuntimeError("Aucune donnée récupérée.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["Code météo (WMO)"] = df_all["Code météo (WMO)"].astype("Int64")

    # FIX 3 : tri sur "Ville" qui existait dans le code original mais dont la colonne était commentée
    df_all = df_all.sort_values(["Date", "Ville"]).reset_index(drop=True)

    return df_all


def apercu(df: pd.DataFrame) -> None:
    """Affiche un aperçu et des statistiques du DataFrame."""
    print("\n── Aperçu (5 premières lignes) ─────────────────────────────────────")
    print(df.head().to_string(index=False))

    print("\n── Types des colonnes ───────────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Statistiques descriptives ────────────────────────────────────────")
    num_cols = ["Température (°C)", "Précipitations (mm)", "Vent (km/h)", "Ensoleillement (MJ/m)"]
    print(df[num_cols].describe().round(2).to_string())

    print("\n── Températures moyennes par pays (ISO) ─────────────────────────────")
    print(
        df.groupby("Pays")["Température (°C)"]
        .mean().round(2)
        .sort_values(ascending=False)
        .to_string()
    )


# Colonnes météo produites par build_dataframe() — déjà au bon nom
COLONNES_METEO = [
    "Température (°C)",
    "Précipitations (mm)",
    "Vent (km/h)",
    "Rafales (km/h)",
    "Ensoleillement (MJ/m)",
]

# Mapping ville → code ISO 3166-1 alpha-2
VILLE_TO_ISO = {
    "Berlin":    "DE",
    "Bruxelles": "BE",
    "Lisbonne":  "PT",
    "Madrid":    "ES",
    "Paris":     "FR",
    "Vienne":    "AT",
}
VILLES_DISPONIBLES = list(VILLE_TO_ISO.keys())


def preproc_meteo(
    df: pd.DataFrame,
    date_start: str,
    date_end: str,
    city: str | list[str],
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
    df["Date"] = pd.to_datetime(df["Date"])

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
        (df["Date"] >= pd.to_datetime(date_start))
        & (df["Date"] <= pd.to_datetime(date_end))
        & (df["Ville"].isin(cities))
    )
    df_filtered = df.loc[mask, ["Date", "Ville"] + COLONNES_METEO].copy()

    if df_filtered.empty:
        raise ValueError(
            f"Aucune donnée trouvée pour la période {date_start} → {date_end} "
            f"et les villes {cities}."
        )

    # Remplacement des noms de villes par leur code ISO 3166-1 alpha-2
    df_filtered["Ville"] = df_filtered["Ville"].map(VILLE_TO_ISO)
    cities_iso = [VILLE_TO_ISO[c] for c in cities]

    df_pivot = df_filtered.pivot(
        index="Date",
        columns="Ville",
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
    df_pivot.rename(columns={"Date": "timestamp"}, inplace=True)
    df_pivot.sort_values("timestamp", inplace=True)
    df_pivot.reset_index(drop=True, inplace=True)

    return df_pivot


# ── Exemple d'utilisation ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # FIX 5 : build_dataframe() était appelé dans le corps de apercu() au lieu du bloc main
    df = build_dataframe()
    apercu(df)

    # Exemples preproc_meteo
    df_paris = preproc_meteo(df, "2022-06-01", "2022-06-30", "Paris")
    print("=== Paris – juin 2022 ===")
    print(df_paris.head())

    df_multi = preproc_meteo(df, "2023-01-01", "2023-01-10", ["Paris", "Madrid", "Berlin"])
    print("\n=== Paris / Madrid / Berlin – 10 premiers jours 2023 ===")
    print(df_multi.to_string())

    df_all = preproc_meteo(df, "2024-07-01", "2024-07-03", "all")
    print("\n=== Toutes les villes – 1-3 juillet 2024 ===")
    print(df_all.to_string())
