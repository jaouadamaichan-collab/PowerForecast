import urllib.request
import urllib.parse
import json
import pandas as pd
import time

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
        f"&timezone=Europe%2FParis"
    )

    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())

    hourly_data = data["hourly"]

    df = pd.DataFrame({
        "Date":                   pd.to_datetime(hourly_data["time"]),
        "Ville":                  city["name"],
        "Pays":                   city["country"],
        "Température (°C)":       hourly_data["temperature_2m"],
        "Précipitations (mm)":    hourly_data["precipitation"],
        "Vent (km/h)":            hourly_data["windspeed_10m"],
        "Rafales (km/h)":         hourly_data["windgusts_10m"],
        "Ensoleillement (MJ/m²)": hourly_data["shortwave_radiation"],
        "Code météo (WMO)":       hourly_data["weathercode"],
    })

    df["Conditions"] = df["Code météo (WMO)"].map(WMO_LABELS).fillna("Inconnu")

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
    df_all["Code météo (WMO)"] = df_all["Code météo (WMO)"].astype("Int64")
    df_all = df_all.sort_values(["Date", "Ville"]).reset_index(drop=True)

    return df_all


def apercu(df: pd.DataFrame) -> None:
    """Affiche un aperçu et des statistiques du DataFrame."""
    print("\n── Aperçu (5 premières lignes) ─────────────────────────────────────")
    print(df.head().to_string(index=False))

    print("\n── Types des colonnes ───────────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Statistiques descriptives ────────────────────────────────────────")
    num_cols = ["Température (°C)", "Précipitations (mm)", "Vent (km/h)", "Ensoleillement (MJ/m²)"]
    print(df[num_cols].describe().round(2).to_string())

    print("\n── Températures moyennes par pays (ISO) ─────────────────────────────")
    print(
        df.groupby("Pays")["Température (°C)"]
        .mean().round(2)
        .sort_values(ascending=False)
        .to_string()
    )


# ── Colonnes météo & mapping ville → ISO ─────────────────────────────────────
COLONNES_METEO = [
    "Température (°C)",
    "Précipitations (mm)",
    "Vent (km/h)",
    "Rafales (km/h)",
    "Ensoleillement (MJ/m²)",
]

VILLE_TO_ISO = {
    'Autriche': 'AUT',
    'Belgique': 'BEL',
    'Bulgarie': 'BGR',
    'Suisse': 'CHE',
    'République tchèque': 'CZE',
    'Allemagne': 'DEU',
    'Danemark': 'DNK',
    'Espagne': 'ESP',
    'Estonie': 'EST',
    'Finlande': 'FIN',
    'France': 'FRA',
    'Grèce': 'GRC',
    'Croatie': 'HRV',
    'Hongrie': 'HUN',
    'Irlande': 'IRL',
    'Italie': 'ITA',
    'Lituanie': 'LTU',
    'Luxembourg': 'LUX',
    'Lettonie': 'LVA',
    'Pays-Bas': 'NLD',
    'Norvège': 'NOR',
    'Pologne': 'POL',
    'Portugal': 'PRT',
    'Roumanie': 'ROU',
    'Serbie': 'SRB',
    'Slovaquie': 'SVK',
    'Slovénie': 'SVN',
    'Suède': 'SWE',
}

VILLES_DISPONIBLES = list(VILLE_TO_ISO.keys())


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


def get_meteo(
    city: str | list,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    Pipeline complet : récupère les données météo et retourne directement
    le DataFrame pivotté prêt à l'emploi.

    Paramètres
    ----------
    city       : Ville(s) parmi VILLES_DISPONIBLES, ou "all".
    date_start : Date de début au format 'YYYY-MM-DD' (incluse).
    date_end   : Date de fin   au format 'YYYY-MM-DD' (incluse).

    Exemple
    -------
    df = get_meteo("Berlin", "2024-01-01", "2024-12-31")
    df = get_meteo(["Berlin", "Paris"], "2024-01-01", "2024-06-30")
    df = get_meteo("all", "2024-01-01", "2024-12-31")
    """
    cities = VILLES_DISPONIBLES if (isinstance(city, str) and city.lower() == "all") else (
        [city] if isinstance(city, str) else list(city)
    )

    invalides = [c for c in cities if c not in VILLES_DISPONIBLES]
    if invalides:
        raise ValueError(
            f"Ville(s) inconnue(s) : {invalides}. "
            f"Choisir parmi : {VILLES_DISPONIBLES}"
        )

    print("Récupération des données météo en cours...\n")
    df_raw = build_dataframe(cities, date_start, date_end)
    df_pivot = preproc_meteo(df_raw, date_start, date_end, city)
    print(f"\n  → DataFrame prêt : {len(df_pivot)} lignes × {len(df_pivot.columns)} colonnes")
    return df_pivot


# ── Paramètres à renseigner ───────────────────────────────────────────────────
df = get_meteo("Berlin", "2024-01-01", "2024-12-31")
