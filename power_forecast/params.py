from pathlib import Path

# ── Paramètres globaux ───────────────────────────────────────────────────────────────
# ── Paths RNN ──────────────────────────────────────────────────────────────



XGB_X_NEW_DATA_DIR  = Path("power_forecast/donnees/x_new_xgb")
XGB_X_NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
XGB_Y_TRUE_DATA_DIR = Path("power_forecast/donnees/y_true_xgb")
XGB_Y_TRUE_DATA_DIR.mkdir(parents=True, exist_ok=True)
RNN_X_NEW_DATA_DIR = Path("power_forecast/donnees/x_new_rnn")
RNN_X_NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
RNN_y_TRUE_DATA_DIR = Path("power_forecast/donnees/y_true_rnn")
RNN_y_TRUE_DATA_DIR.mkdir(parents=True, exist_ok=True)


INPUT_LENGTH = 21 * 24  # 3 weeks context fed to RNN
INPUT_LENGTH_RNN  = INPUT_LENGTH
OUTPUT_LENGTH = 48  # predict 24h of target day
HORIZON = 0  # skip 24h between input end and output

VALID_STEPS = {"h", "D"}

# Outliers limits of electricity prices
LIMIT_LOW = -350
LIMIT_HIGH = 900

# in hours, corresponds to 2 semaines (même heure)

# XGB Lags must be higher than 24h x target distance to avoid leakage, and not too high to keep some signal.
# We keep 48h, 72h and 168h (1 week) for the neighboring countries, and add more lags for the target country itself since it has more signal.
LAGS_XGB_TARGET = [
    48,  # same hour 2 days ago, to capture the weekly seasonality and the target distance
    49,  # 1 hour before the same hour 2 days ago, to capture the hourly seasonality and the target distance
    50,  # 2 hours before the same hour 2 days ago, to capture the hourly seasonality and the target distance
    51,
    54,
    60,
    72,
    168,
]

MAX_LAG_BACK_XGB = LAGS_XGB_TARGET[-1]

ROLLING_WINDOWS_XGB_TARGET = [6, 12, 24, 48, 62, 96]

LAGS_XGB_FRONTIERE = [
    48,  # same hour 2 days ago, to capture the weekly seasonality and the target distance
    49,  # 1 hour before the same hour 2 days ago, to capture the hourly seasonality and the target distance
    72,
]

ROLLING_WINDOWS_XGB_FRONTIERE = [
    12,
    24,
    48,
] # every rollwing windows compute two values for each country





DROP_COLUMN_NAN_TRESHOLD = 0.05  # Drop columns with more than 5% NaN


ROLLING_WINDOWS_TARGET = [6, 12, 24, 48, 62, 120]  # en heures

ROLLING_WINDOWS_FRONTIERE = [48, 62, 120]  # en heuresGB


# ISO codes of countries to predict (must be in ENTSOE and COUNTRY_LABELS)
VALID = [
    "AUT",
    "BEL",
    "BGR",
    "CHE",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GRC",
    "HRV",
    "HUN",
    "IRL",
    "ITA",
    "LTU",
    "LUX",
    "LVA",
    "NLD",
    "NOR",
    "POL",
    "PRT",
    "ROU",
    "SRB",
    "SVK",
    "SVN",
    "SWE",
]

FRONTIERE = {
    "AUT": ["CHE", "DEU", "CZE", "SVK", "HUN", "SVN", "ITA"],
    "BEL": ["FRA", "LUX", "DEU", "NLD"],
    "BGR": ["ROU", "SRB", "GRC"],
    "CHE": ["FRA", "DEU", "AUT", "ITA"],
    "CZE": ["DEU", "POL", "SVK", "AUT"],
    "DEU": [
        "DNK",
        "NLD",
        "BEL",
        "LUX",
        "FRA",
        "CHE",
        "AUT",
        "CZE",
        "POL",
        "SWE",
        "NOR",
    ],
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

OBJ_ISO_TO_ENTSOE = {
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "CHE": "CH",
    "CZE": "CZ",
    "DEU": "DE_LU",
    "DNK": "DK",
    "ESP": "ES",
    "EST": None,  # not in COUNTRY_LABELS
    "FIN": "FI",
    "FRA": "FR",
    "GRC": "GR",
    "HRV": "HR",
    "HUN": "HU",
    "IRL": None,  # not in COUNTRY_LABELS
    "ITA": "IT",
    "LTU": None,  # not in COUNTRY_LABELS
    "LUX": "DE_LU",  # merged into DE_LU
    "LVA": None,  # not in COUNTRY_LABELS
    "NLD": "NL",
    "NOR": "NO",
    "POL": "PL",
    "PRT": "PT",
    "ROU": "RO",
    "SRB": None,  # not in COUNTRY_LABELS
    "SVK": "SK",
    "SVN": "SI",
    "SWE": "SE",
}

# filter to be safe
FRONTIERE = {
    k: list(dict.fromkeys([n for n in v if n in VALID]))  # deduplicate + filter
    for k, v in FRONTIERE.items()
    if k in VALID
}


# Manual crisis periods to add as features (list of tuples with start and end dates)
CRISIS_PERIODS = [
    ("2015-01-01", "2016-06-30"),
    ("2020-03-01", "2020-12-31"),
    ("2021-06-01", "2023-06-30"),
    ("2023-10-01", "2024-03-31"),
]


# Params catch22
WINDOW_CATCH22 = 7  # in days, for hourly data this means 168 hours
STEP_CATCH22 = 1  # in days, for hourly data this means
TIMESTAMP_CATCH22 = 'h'

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
    "temperature_c",
    "precipitation_mm",
    "vent_km_h",
    "rafales_km_h",
    "irradiation_MJ_m2",
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
DEFAULT_API_KEY = "eae53aee-610c-46a9-8bce-f350cfb45736"
DEFAULT_TIMEZONE = "UTC"
DEFAULT_START = "2025-01-01"
DEFAULT_END = "2025-06-30"
DEFAULT_STEP = "H"  # "H" ou "D"
# DEFAULT_COUNTRIES = ["FR", "DE_LU", "ES", "BE"]
# DEFAULT_TARGET_COUNTRY = "FR"

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


_DONNEES_DIR = Path(__file__).parent / "donnees"

LOCAL_DATA_PATH_MODELS = str(_DONNEES_DIR / "saved_models")
LOCAL_REGISTRY_PATH_MODELS = str(_DONNEES_DIR / "saved_models")

LOCAL_DATA_PATH_DF = str(_DONNEES_DIR / "df")
LOCAL_REGISTRY_PATH_DF = str(_DONNEES_DIR / "df")

LOCAL_DATA_PATH_SC = str(_DONNEES_DIR / "scaler")
LOCAL_REGISTRY_PATH_SC = str(_DONNEES_DIR / "scaler")



PICKLE_DIR = Path("raw_data/pickle_files")
METEO_CACHE_DIR = PICKLE_DIR / "meteo_cache"  # ← build from PICKLE_DIR, don't hardcode
METEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


LAGS_TARGET = [
    48,
    49,
    72,  # 2-3 jours
    168,
    336,  # 1, 2 semaines (même heure)
]

LAGS_FRONTIERE = [
    48,
    72,  # 2-3 jours
    168,
]

GCS_BUCKET='power_forecast_bucket'
GOOGLE_CLOUD_PROJECT='bootcamp-wagon-2195'
