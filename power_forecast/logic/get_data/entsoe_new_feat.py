# ══════════════════════════════════════════════════════════════════════════════
# Nouvelles fonctions de récupération de données
# ══════════════════════════════════════════════════════════════════════════════

# Install entsoe-py if not already installed
try:
    from entsoe import EntsoePandasClient
    import pandas as pd
except ImportError:
    !pip install entsoe-py pandas matplotlib
    from entsoe import EntsoePandasClient
    import pandas as pd

def fetch_actual_generation(client: EntsoePandasClient, country: str, start: pd.Timestamp, end: pd.Timestamp, step: str = DEFAULT_STEP) -> pd.DataFrame:
    """
    Récupère la production réelle par type pour un pays.
    Renvoie un DataFrame avec les colonnes pour différents types de production.
    """
    log.info("⬇  Récupération de la production réelle pour %s (%s → %s)…",
             COUNTRY_LABELS.get(country, country), start.date(), end.date())

    # Common production types requested by user and generally available.
    # Using B07 for Nuclear, B11 for Wind Onshore, B12 for Solar
    production_types = {
        'Nuclear': 'B07',
        'Wind Onshore': 'B11',
        'Solar': 'B12'
    }

    all_generation_data = {}
    for p_type_name, p_type_code in production_types.items():
        try:
            raw_gen = safe_fetch(client.query_actual_generation_per_production_type, country, start=start, end=end, psr_type=p_type_code)
            if not raw_gen.empty: # Check if data was actually returned
                # The entsoe-py client returns a multi-indexed series, we want to extract the generation values
                # It might return for different units. Let's simplify by summing up if multiple units
                if isinstance(raw_gen.index, pd.MultiIndex):
                    # Attempt to sum across the second level if it exists and contains 'Actual Generation'
                    if 'Actual Generation' in raw_gen.index.get_level_values(1):
                        series = raw_gen.xs('Actual Generation', level=1)
                    else:
                        # If 'Actual Generation' not found, take the first available type or sum all
                        series = raw_gen.groupby(level=0).sum()
                else: # Assuming it's already a single series of generation data
                    series = raw_gen

                all_generation_data[p_type_name] = resample_series(series, step)
                log.info("   ✓ %d points de production %s récupérés (pas : %s)", len(series), p_type_name, step_label(step))
            else:
                log.warning("   Pas de données pour la production %s pour %s.", p_type_name, country)
        except Exception as exc:
            log.error("❌  Impossible de récupérer les données de production %s pour %s : %s", p_type_name, country, exc)

    if not all_generation_data:
        log.warning("Aucune donnée de production récupérée pour %s.", country)
        return pd.DataFrame()

    df_gen = pd.DataFrame(all_generation_data)
    df_gen.columns = [f'{col}_{country}' for col in df_gen.columns] # Prefix with country code
    return df_gen

def fetch_actual_load(client: EntsoePandasClient, country: str, start: pd.Timestamp, end: pd.Timestamp, step: str = DEFAULT_STEP) -> pd.Series:
    """
    Récupère la consommation réelle (charge) pour un pays.
    """
    log.info("⬇  Récupération de la consommation réelle pour %s (%s → %s)…",
             COUNTRY_LABELS.get(country, country), start.date(), end.date())
    raw_load = safe_fetch(client.query_actual_load, country, start=start, end=end)
    load_data = resample_series(raw_load, step)
    log.info("   ✓ %d points de consommation récupérés (pas : %s)", len(load_data), step_label(step))
    load_data.name = f'ActualLoad_{country}'
    return load_data

def fetch_wind_solar_forecast(client: EntsoePandasClient, country: str, start: pd.Timestamp, end: pd.Timestamp, step: str = DEFAULT_STEP) -> pd.DataFrame:
    """
    Récupère les prévisions de production éolienne et solaire pour un pays.
    Renvoie un DataFrame avec les colonnes pour les prévisions éoliennes et solaires.
    """
    log.info("⬇  Récupération des prévisions éoliennes et solaires pour %s (%s → %s)…",
             COUNTRY_LABELS.get(country, country), start.date(), end.date())

    forecast_types = {
        'WindForecast': 'B11', # Wind Onshore
        'SolarForecast': 'B12' # Solar
    }

    all_forecast_data = {}
    for f_type_name, f_type_code in forecast_types.items():
        try:
            raw_forecast = safe_fetch(client.query_wind_and_solar_forecast, country, start=start, end=end, psr_type=f_type_code)
            if not raw_forecast.empty: # Check if data was actually returned
                # The entsoe-py client returns a multi-indexed series, take 'Actual Aggregated' or similar if available
                if isinstance(raw_forecast.index, pd.MultiIndex) and 'Actual Aggregated' in raw_forecast.index.get_level_values(1):
                    series = raw_forecast.xs('Actual Aggregated', level=1)
                else:
                    series = raw_forecast
                all_forecast_data[f_type_name] = resample_series(series, step)
                log.info("   ✓ %d points de prévision %s récupérés (pas : %s)", len(series), f_type_name, step_label(step))
            else:
                log.warning("   Pas de données pour les prévisions %s pour %s.", f_type_name, country)
        except Exception as exc:
            log.error("❌  Impossible de récupérer les prévisions %s pour %s : %s", f_type_name, country, exc)

    if not all_forecast_data:
        log.warning("Aucune donnée de prévision éolienne/solaire récupérée pour %s.", country)
        return pd.DataFrame()

    df_forecast = pd.DataFrame(all_forecast_data)
    df_forecast.columns = [f'{col}_{country}' for col in df_forecast.columns]
    return df_forecast


