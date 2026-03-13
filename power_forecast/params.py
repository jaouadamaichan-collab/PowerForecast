import os

#Params catch22
WINDOW_CATCH22 = 7  # in days, for hourly data this means 168 hours
STEP_CATCH22 = 1  # in days, for hourly data this means

# Outliers limits
LIMIT_LOW = -350
LIMIT_HIGH = 900

LAGS_TARGET = [
    1,
    2,
    3,  # court terme
    6,
    12,
    24,  # intra-journalier
    48,
    72,  # 2-3 jours
    168,
    336,  # 1, 2 semaines (même heure)
]

LAGS_FRONTIERE = [
    48,
    72,  # 2-3 jours
    168,
]

ROLLING_WINDOWS_TARGET = [6, 12, 24, 48, 62, 120]  # en heures

ROLLING_WINDOWS_FRONTIERE = [48, 62, 120]  # en heuresGB

VALID = ['AUT', 'BEL', 'BGR', 'CHE', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 
         'FRA', 'GRC', 'HRV', 'HUN', 'IRL', 'ITA', 'LTU', 'LUX', 'LVA', 'NLD', 
         'NOR', 'POL', 'PRT', 'ROU', 'SRB', 'SVK', 'SVN', 'SWE']

FRONTIERE = {
    "AUT": ["CHE", "DEU", "CZE", "SVK", "HUN", "SVN", "ITA"],
    "BEL": ["FRA", "LUX", "DEU", "NLD"],
    "BGR": ["ROU", "SRB", "GRC"],
    "CHE": ["FRA", "DEU", "AUT", "ITA"],
    "CZE": ["DEU", "POL", "SVK", "AUT"],
    "DEU": ["DNK", "NLD", "BEL", "LUX", "FRA", "CHE", "AUT", "CZE", "POL", "SWE", "NOR"],
    "DNK": ["DEU", "SWE", "NOR", "NLD"],
    "ESP": ["FRA", "PRT"],
    "EST": ["LVA", "LTU", "FIN"],
    "FIN": ["SWE", "NOR", "EST"],
    "FRA": ["BEL", "LUX", "DEU", "CHE", "ITA", "ESP", "IRL"],
    "GRC": ["BGR", "ITA"],
    "HRV": ["SVN", "HUN", "SRB"],
    "HUN": ["AUT", "SVK", "SVN", "HRV", "ROU", "SRB"],
    "IRL": ["FRA"],
    "ITA": ["FRA", "CHE", "AUT", "SVN", "GRC"],
    "LTU": ["LVA", "POL", "EST", "SWE"],
    "LUX": ["BEL", "FRA", "DEU"],
    "LVA": ["EST", "LTU"],
    "NLD": ["BEL", "DEU", "DNK", "NOR"],
    "NOR": ["SWE", "FIN", "DEU", "NLD", "DNK"],
    "POL": ["DEU", "CZE", "SVK", "LTU", "SWE"],
    "PRT": ["ESP"],
    "ROU": ["HUN", "BGR", "SRB"],
    "SRB": ["HUN", "ROU", "BGR", "HRV"],
    "SVK": ["CZE", "AUT", "HUN", "POL"],
    "SVN": ["ITA", "AUT", "HUN", "HRV"],
    "SWE": ["NOR", "FIN", "DNK", "DEU", "LTU", "POL"],
}

# filter to be safe
FRONTIERE = {
    k: list(dict.fromkeys([n for n in v if n in VALID]))  # deduplicate + filter
    for k, v in FRONTIERE.items()
    if k in VALID
}


CRISIS_PERIODS = [
    ("2015-01-01", "2016-06-30"),
    ("2020-03-01", "2020-12-31"),
    ("2021-06-01", "2023-06-30"),
    ("2023-10-01", "2024-03-31"),
]


# Variables Météo

WMO_LABELS = {
    0: "Ciel dégagé",
    1: "Principalement dégagé",
    2: "Partiellement nuageux",
    3: "Couvert",
    45: "Brouillard",
    48: "Brouillard givrant",
    51: "Bruine légère",
    53: "Bruine modérée",
    55: "Bruine dense",
    61: "Pluie légère",
    63: "Pluie modérée",
    65: "Pluie forte",
    71: "Neige légère",
    73: "Neige modérée",
    75: "Neige forte",
    80: "Averses légères",
    81: "Averses modérées",
    82: "Averses violentes",
    95: "Orage",
    96: "Orage avec grêle",
    99: "Orage avec forte grêle",
}

COLONNES_METEO = [
    "Température (°C)",
    "Précipitations (mm)",
    "Vent (km/h)",
    "Rafales (km/h)",
    "Ensoleillement (MJ/m²)",
]

VILLE_TO_ISO = {
    "Autriche": "AUT",
    "Belgique": "BEL",
    "Bulgarie": "BGR",
    "Suisse": "CHE",
    "République tchèque": "CZE",
    "Allemagne": "DEU",
    "Danemark": "DNK",
    "Espagne": "ESP",
    "Estonie": "EST",
    "Finlande": "FIN",
    "France": "FRA",
    "Grèce": "GRC",
    "Croatie": "HRV",
    "Hongrie": "HUN",
    "Irlande": "IRL",
    "Italie": "ITA",
    "Lituanie": "LTU",
    "Luxembourg": "LUX",
    "Lettonie": "LVA",
    "Pays-Bas": "NLD",
    "Norvège": "NOR",
    "Pologne": "POL",
    "Portugal": "PRT",
    "Roumanie": "ROU",
    "Serbie": "SRB",
    "Slovaquie": "SVK",
    "Slovénie": "SVN",
    "Suède": "SWE",
}
VILLES_DISPONIBLES = list(VILLE_TO_ISO.keys())

# Variables ENTSOE
PALETTE = [
    "#2196F3",
    "#E91E63",
    "#4CAF50",
    "#FF9800",
    "#9C27B0",
    "#00BCD4",
    "#FF5722",
    "#607D8B",
    "#795548",
    "#009688",
]

# ── Constantes ─────────────────────────────────────────────────────────────────
DEFAULT_API_KEY = "82e51d13-ad8c-44f1-99bd-a000cbcc2ae3"
DEFAULT_TIMEZONE = "Europe/Paris"
DEFAULT_START = "2025-01-01"
DEFAULT_END = "2025-06-30"
DEFAULT_STEP = "D"  # "H" ou "D"
DEFAULT_COUNTRIES = ["FR"]

# Catalogue des zones disponibles (code ENTSO-E → label lisible)
COUNTRY_LABELS = {
    "FR": "France",
    "DE_LU": "Allemagne / Luxembourg",
    "ES": "Espagne",
    "BE": "Belgique",
    "NL": "Pays-Bas",
    "IT": "Italie",
    "CH": "Suisse",
    "AT": "Autriche",
    "PT": "Portugal",
    "PL": "Pologne",
    "CZ": "République Tchèque",
    "SK": "Slovaquie",
    "HU": "Hongrie",
    "RO": "Roumanie",
    "BG": "Bulgarie",
    "GR": "Grèce",
    "HR": "Croatie",
    "SI": "Slovénie",
    "SE": "Suède",
    "NO": "Norvège",
    "DK": "Danemark",
    "FI": "Finlande",
    "GB": "Royaume-Uni",
}


# Save parquet
PICKLE_DIR = "raw_data/pickle_files"


COUNTRY_HOLIDAY_MAP = {
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "CHE": "CH",
    "CZE": "CZ",
    "DEU": "DE",
    "DNK": "DK",
    "ESP": "ES",
    "EST": "EE",
    "FIN": "FI",
    "FRA": "FR",
    "GRC": "GR",
    "HRV": "HR",
    "HUN": "HU",
    "IRL": "IE",
    "ITA": "IT",
    "LTU": "LT",
    "LUX": "LU",
    "LVA": "LV",
    "NLD": "NL",
    "NOR": "NO",
    "POL": "PL",
    "PRT": "PT",
    "ROU": "RO",
    "SRB": "RS",
    "SVK": "SK",
    "SVN": "SI",
    "SWE": "SE",
}

LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "models")
