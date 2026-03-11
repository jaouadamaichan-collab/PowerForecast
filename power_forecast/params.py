#Params catch22
WINDOW_CATCH22 = 7  # in days, for hourly data this means 168 hours
STEP_CATCH22   = 1  # in days, for hourly data this means

#Outliers limits
LIMIT_LOW = -350
LIMIT_HIGH = 3200

# Variables Météo

WMO_LABELS = {
    0: "Ciel dégagé", 1: "Principalement dégagé", 2: "Partiellement nuageux",
    3: "Couvert", 45: "Brouillard", 48: "Brouillard givrant",
    51: "Bruine légère", 53: "Bruine modérée", 55: "Bruine dense",
    61: "Pluie légère", 63: "Pluie modérée", 65: "Pluie forte",
    71: "Neige légère", 73: "Neige modérée", 75: "Neige forte",
    80: "Averses légères", 81: "Averses modérées", 82: "Averses violentes",
    95: "Orage", 96: "Orage avec grêle", 99: "Orage avec forte grêle",
}

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

# Variables ENTSOE
PALETTE = [
    "#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#607D8B", "#795548", "#009688",
]

# ── Constantes ─────────────────────────────────────────────────────────────────
DEFAULT_API_KEY = "82e51d13-ad8c-44f1-99bd-a000cbcc2ae3"
DEFAULT_TIMEZONE = "Europe/Paris"
DEFAULT_START    = "2025-01-01"
DEFAULT_END      = "2025-06-30"
DEFAULT_STEP     = "D"          # "H" ou "D"
DEFAULT_COUNTRIES = ["FR"]

# Catalogue des zones disponibles (code ENTSO-E → label lisible)
COUNTRY_LABELS = {
    "FR":    "France",
    "DE_LU": "Allemagne / Luxembourg",
    "ES":    "Espagne",
    "BE":    "Belgique",
    "NL":    "Pays-Bas",
    "IT":    "Italie",
    "CH":    "Suisse",
    "AT":    "Autriche",
    "PT":    "Portugal",
    "PL":    "Pologne",
    "CZ":    "République Tchèque",
    "SK":    "Slovaquie",
    "HU":    "Hongrie",
    "RO":    "Roumanie",
    "BG":    "Bulgarie",
    "GR":    "Grèce",
    "HR":    "Croatie",
    "SI":    "Slovénie",
    "SE":    "Suède",
    "NO":    "Norvège",
    "DK":    "Danemark",
    "FI":    "Finlande",
    "GB":    "Royaume-Uni",
}





# Save parquet
PICKLE_DIR = "raw_data/pickle_files"
