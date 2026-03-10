import urllib.request
import json
import pandas as pd
from datetime import datetime
import time # Import the time module

# ── Villes européennes (ajouter/supprimer selon besoin) ──────────────────────
CITIES = [
    {"name": "Berlin",     "country": "Allemagne",   "lat": 52.5200,  "lon": 13.4050},
]

# "name": "Paris",      "country": "France",      "lat": 48.8566,  "lon":  2.3522
# "name": "Berlin",     "country": "Allemagne",   "lat": 52.5200,  "lon": 13.4050
# "name": "Madrid",     "country": "Espagne",     "lat": 40.4168,  "lon": -3.7038
# "name": "Rome",       "country": "Italie",      "lat": 41.9028,  "lon": 12.4964
# "name": "Amsterdam",  "country": "Pays-Bas",    "lat": 52.3676,  "lon":  4.9041
# "name": "Bruxelles",  "country": "Belgique",    "lat": 50.8503,  "lon":  4.3517
# "name": "Vienne",     "country": "Autriche",    "lat": 48.2082,  "lon": 16.3738
# "name": "Lisbonne",   "country": "Portugal",    "lat": 38.7223,  "lon": -9.1393
# "name": "Stockholm",  "country": "Suède",       "lat": 59.3293,  "lon": 18.0686
# "name": "Varsovie",   "country": "Pologne",     "lat": 52.2297,  "lon": 21.0122

# ── Sélection active : changer ici pour basculer entre les deux listes ────────
CITIES_ACTIVE = CITIES                    # ← capitales européennes


# ── Plage de dates (format YYYY-MM-DD) ───────────────────────────────────────
DATE_DEBUT = "2015-01-01"   # ← modifier ici
DATE_FIN   = "2024-12-31"   # ← modifier ici

# ── Fichier de sortie ─────────────────────────────────────────────────────────
OUTPUT_FILE = "meteo_historique_europe.csv"

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
    # Request hourly instantaneous variables
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

    # Construire le DataFrame directement depuis le JSON avec des noms de colonnes adaptés à l'heure
    df = pd.DataFrame({
        "Date":                   pd.to_datetime(hourly_data["time"]),
        "Ville":                  city["name"],
        "Pays":                   city["country"],
        "Région":                 city.get("region", ""),
        "Latitude":               city["lat"],
        "Longitude":              city["lon"],
        "Température (°C)":       hourly_data["temperature_2m"],
        "Précipitations (mm)":    hourly_data["precipitation"],
        "Vent (km/h)":            hourly_data["windspeed_10m"],
        "Rafales (km/h)":         hourly_data["windgusts_10m"],
        "Direction_vent ()":     hourly_data["winddirection_10m"],
        "Ensoleillement (MJ/m)": hourly_data["shortwave_radiation"],
        "Code météo (WMO)":       hourly_data["weathercode"],
    })

    # Colonne conditions lisibles
    df["Conditions"] = df["Code météo (WMO)"].map(WMO_LABELS).fillna("Inconnu")

    return df


def build_dataframe() -> pd.DataFrame:
    """Agrège les données de toutes les villes en un seul DataFrame."""
    frames = []
    for city in CITIES_ACTIVE:
        try:
            df = fetch_historical(city, DATE_DEBUT, DATE_FIN)
            frames.append(df)
            # Adjust message to reflect hourly data fetched (24 times days)
            print(f"  ✓ {city['name']} ({city['country']}) — {len(df) // 24} jours (soit {len(df)} heures)")
        except Exception as e:
            print(f"  ✘ {city['name']} — Erreur : {e}")
        time.sleep(10) # Add a 1-second delay between API calls

    if not frames:
        raise RuntimeError("Aucune donnée récupérée.")

    # Concaténation + typage
    df_all = pd.concat(frames, ignore_index=True)
    df_all["Code météo (WMO)"] = df_all["Code météo (WMO)"].astype("Int64")

    # Tri par date et ville
    df_all = df_all.sort_values(["Date", "Ville"]).reset_index(drop=True)

    return df_all


def apercu(df: pd.DataFrame) -> None:
    """Affiche un aperçu et des statistiques du DataFrame."""
    print("\n── Aperçu (5 premières lignes) ─────────────────────────────────────")
    print(df.head().to_string(index=False))

    print("\n── Types des colonnes ───────────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Statistiques descriptives ────────────────────────────────────────")
    # Update num_cols to reflect new hourly column names
    num_cols = ["Température (°C)",
                "Précipitations (mm)", "Vent (km/h)", "Ensoleillement (MJ/m)"]
    print(df[num_cols].describe().round(2).to_string())

    print("\n── Températures moyennes par pays ───────────────────────────────────")
    print(
        df.groupby("Pays")["Température (°C)"] # Use new temperature column
        .mean().round(2)
        .sort_values(ascending=False)
        .to_string()
    )


def main():
    # Validation des dates
    try:
        d1 = datetime.strptime(DATE_DEBUT, "%Y-%m-%d")
        d2 = datetime.strptime(DATE_FIN,   "%Y-%m-%d")
    except ValueError:
        print("❌ Format de date invalide. Utilisez YYYY-MM-DD.")
        return
    if d1 > d2:
        print("❌ DATE_DEBUT doit être antérieure à DATE_FIN.")
        return

    print(f"Période : {DATE_DEBUT} → {DATE_FIN}")
    print(f"Villes  : {len(CITIES_ACTIVE)}\n")

    # ── Construction du DataFrame ─────────────────────────────────────────────
    df = build_dataframe()

    # ── Aperçu console ────────────────────────────────────────────────────────
    apercu(df)

    # ── Export CSV ────────────────────────────────────────────────────────────
    # df.to_csv(OUTPUT_FILE, index=False, sep=";", encoding="utf-8-sig",
    #           date_format="%Y-%m-%d %H:00:00") # Changed date_format for hourly

    # print(f"\n✅ DataFrame exporté : {OUTPUT_FILE}")
    # print(f"   {len(df)} lignes × {len(df.columns)} colonnes")

    return df   # utile en notebook Jupyter


if __name__ == "__main__":
    df = main()
