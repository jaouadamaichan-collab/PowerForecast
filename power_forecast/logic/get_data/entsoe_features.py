"""
╔══════════════════════════════════════════════════════════════╗
║         ENTSO-E — Récupération automatisée des prix          ║
║         Day-Ahead pour un ou plusieurs pays                  ║
╚══════════════════════════════════════════════════════════════╝

Usage :
    python entsoe_prices_fetcher.py
    python entsoe_prices_fetcher.py --countries FR DE_LU ES --target FR --start 2025-01-01 --end 2025-06-30
    python entsoe_prices_fetcher.py --countries FR DE_LU --target DE_LU --step H --no-plot
    python entsoe_prices_fetcher.py --list-countries

    --countries : liste de pays pour les prix day-ahead (comparaison multi-pays)
    --target    : pays cible pour les données détaillées (production, consommation,
                  prévisions éolien/solaire). Doit faire partie de --countries.
                  Si omis, utilise DEFAULT_TARGET_COUNTRY.

Installation :
    pip install entsoe-py pandas matplotlib
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import logging
from pathlib import Path
# from power_forecast.params import * # Parameters are now defined in cell 429293e2

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from power_forecast.params import *

try:
    from entsoe import EntsoePandasClient
except ImportError:
    print("❌  Bibliothèque manquante. Lancez : pip install entsoe-py")
    sys.exit(1)

# ── Configuration logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("entsoe_fetcher")



def fetch_actual_generation(
    client: EntsoePandasClient,
    country: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    step: str = DEFAULT_STEP,
) -> pd.DataFrame:
    """
    Récupère la production réelle par type d'énergie pour un pays.

    Returns
    -------
    pd.DataFrame — index DatetimeTZ, colonnes par type de production (MW)
    """
    log.info("⬇  Récupération de la production pour %s (%s → %s)…",
             COUNTRY_LABELS.get(country, country), start.date(), end.date())
    try:
        raw = safe_fetch(client.query_generation, country, start=start, end=end)
        if isinstance(raw, pd.DataFrame):
            raw.columns = [
                f"{country}_{col}" if isinstance(col, str) else f"{country}_{col[0]}"
                for col in raw.columns
            ]
        result = resample_series(raw, step)
        log.info("   ✓ Production récupérée (%d colonnes)",
                 result.shape[1] if isinstance(result, pd.DataFrame) else 1)
        return result
    except Exception as exc:
        log.error("❌  Impossible de récupérer la production pour %s : %s", country, exc)
        return pd.DataFrame()


def fetch_actual_load(
    client: EntsoePandasClient,
    country: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    step: str = DEFAULT_STEP,
) -> pd.Series:
    """
    Récupère la consommation réelle pour un pays.

    Returns
    -------
    pd.Series — index DatetimeTZ, valeurs en MW
    """
    log.info("⬇  Récupération de la consommation pour %s (%s → %s)…",
             COUNTRY_LABELS.get(country, country), start.date(), end.date())
    try:
        raw    = safe_fetch(client.query_load, country, start=start, end=end)
        result = resample_series(raw, step)
        result.name = country
        log.info("   ✓ Consommation récupérée (%d points)", len(result))
        return result
    except Exception as exc:
        log.error("❌  Impossible de récupérer la consommation pour %s : %s", country, exc)
        return pd.Series(dtype=float)


def fetch_wind_solar_forecast(
    client: EntsoePandasClient,
    country: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    step: str = DEFAULT_STEP,
) -> pd.DataFrame:
    """
    Récupère les prévisions de production éolienne et solaire pour un pays.

    Returns
    -------
    pd.DataFrame — index DatetimeTZ, colonnes Wind et Solar (MW)
    """
    log.info("⬇  Récupération des prévisions éolien/solaire pour %s (%s → %s)…",
             COUNTRY_LABELS.get(country, country), start.date(), end.date())
    try:
        raw = safe_fetch(client.query_wind_and_solar_forecast, country, start=start, end=end)
        if isinstance(raw, pd.DataFrame):
            raw.columns = [f"{country}_{col}" for col in raw.columns]
        result = resample_series(raw, step)
        log.info("   ✓ Prévisions éolien/solaire récupérées (%d colonnes)",
                 result.shape[1] if isinstance(result, pd.DataFrame) else 1)
        return result
    except Exception as exc:
        log.error("❌  Impossible de récupérer les prévisions éolien/solaire pour %s : %s", country, exc)
        return pd.DataFrame()


def get_gen_load_forecast(
    target: str,
    date_start: str,
    date_end: str,
    step: str = "H",
    api_key: str = DEFAULT_API_KEY,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:

    if step not in VALID_STEPS:
        raise ValueError(f"Paramètre 'step' invalide : '{step}'. Valeurs acceptées : {VALID_STEPS}")

    client   = EntsoePandasClient(api_key=api_key)
    ts_start = parse_date(date_start, timezone)
    ts_end   = parse_date(date_end,   timezone)
    log.info("═" * 60)
    log.info("Données détaillées   : %s", COUNTRY_LABELS.get(target, target))
    log.info("Période              : %s → %s | Pas : %s",
             ts_start.date(), ts_end.date(), step_label(step))
    log.info("═" * 60)
    df_generation  = fetch_actual_generation(client, target, ts_start, ts_end, step=step)
    df_load_series = fetch_actual_load(client, target, ts_start, ts_end, step=step)
    df_forecast    = fetch_wind_solar_forecast(client, target, ts_start, ts_end, step=step)

    df_load = (
        df_load_series.to_frame(name=f"{target}_load")
        if isinstance(df_load_series, pd.Series)
        else df_load_series
    )

    frames = [df_generation, df_load, df_forecast]
    frames = [f[~f.index.duplicated(keep='first')] for f in frames]
    df = pd.concat(frames, axis=1)
    df.index.name = "timestamp"
    print(f"  ✓ Gen/Load/Forecast : {df.shape}")
    return df

def get_gen_load_forecast(
    target: str,
    start: str,
    end: str,
    step: str = "D",
    api_key: str = DEFAULT_API_KEY,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """
    Récupère les données détaillées (production, consommation, prévisions éolien/solaire)
    pour un pays cible et retourne un DataFrame fusionné.

    Parameters
    ----------
    target : str
        Pays cible. Ex : "FR"
    start : str
        Date de début "YYYY-MM-DD".
    end : str
        Date de fin "YYYY-MM-DD".
    step : str, optional
        "H" (horaire) ou "D" (journalier, défaut).
    api_key : str, optional
        Clé API ENTSO-E.
    timezone : str, optional
        Fuseau horaire (défaut : "Europe/Paris").

    Returns
    -------
    pd.DataFrame
        DataFrame fusionné avec toutes les colonnes sur l'index temporel commun.

    Examples
    --------
    df = get_full_data(target="FR", start="2025-01-01", end="2025-06-30", step="H")
    """
    if step not in VALID_STEPS:
        raise ValueError(f"Paramètre 'step' invalide : '{step}'. Valeurs acceptées : {VALID_STEPS}")

    if target not in COUNTRY_LABELS:
        log.warning("Code pays inconnu : %s — vérifiez la liste COUNTRY_LABELS.", target)

    client   = EntsoePandasClient(api_key=api_key)
    ts_start = parse_date(start, timezone)
    ts_end   = parse_date(end,   timezone)

    log.info("═" * 60)
    log.info("Données détaillées   : %s", COUNTRY_LABELS.get(target, target))
    log.info("Période              : %s → %s | Pas : %s",
             ts_start.date(), ts_end.date(), step_label(step))
    log.info("═" * 60)

    df_generation  = fetch_actual_generation(client, target, ts_start, ts_end, step=step)
    df_load_series = fetch_actual_load(client, target, ts_start, ts_end, step=step)
    df_forecast    = fetch_wind_solar_forecast(client, target, ts_start, ts_end, step=step)

    df_load = (
        df_load_series.to_frame(name=f"{target}_load")
        if isinstance(df_load_series, pd.Series)
        else df_load_series
    )

    frames = [df_generation, df_load, df_forecast]
    frames = [f[~f.index.duplicated(keep='first')] for f in frames]

    df = pd.concat(frames, axis=1)
    df.index.name = "timestamp"

    return df



def get_all_prices(
    countries: str | list[str],
    start: str,
    end: str,
    step: str = "D",
    api_key: str = DEFAULT_API_KEY,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    
    if step not in VALID_STEPS:
        raise ValueError(f"Paramètre 'step' invalide : '{step}'. Valeurs acceptées : {VALID_STEPS}")

    country_list = [countries] if isinstance(countries, str) else list(countries)

    unknown = [c for c in country_list if c not in COUNTRY_LABELS]
    if unknown:
        log.warning("Code(s) pays inconnu(s) : %s", unknown)

    client   = EntsoePandasClient(api_key=api_key)
    ts_start = parse_date(start, timezone)
    ts_end   = parse_date(end,   timezone)

    frames = {}
    for country in country_list:
        try:
            frames[country] = fetch_prices(client, country, ts_start, ts_end, step)
        except Exception as exc:
            log.error("❌ Prix indisponibles pour %s : %s", country, exc)

    if not frames:
        raise RuntimeError("Aucun prix récupéré. Vérifiez la clé API et les codes pays.")

    df = pd.DataFrame(frames)
    df.index.name = "timestamp"
    return df


def fetch_prices(
    client: EntsoePandasClient,
    country: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    step: str = DEFAULT_STEP,
) -> pd.Series:
    """
    Récupère et rééchantillonne les prix day-ahead pour un pays.

    Returns
    -------
    pd.Series  — index DatetimeTZ, valeurs en €/MWh
    """
    if step not in VALID_STEPS:
        raise ValueError(f"Paramètre 'step' invalide : '{step}'. Valeurs acceptées : {VALID_STEPS}")

    log.info("⬇  Récupération des prix pour %s (%s → %s)…",
             COUNTRY_LABELS.get(country, country), start.date(), end.date())

    raw    = safe_fetch(client.query_day_ahead_prices, country, start=start, end=end)
    prices = resample_series(raw, step)

    log.info("   ✓ %d points récupérés (pas : %s)", len(prices), step_label(step))
    return prices


def parse_date(date_str: str, timezone: str) -> pd.Timestamp:
    """Convertit une chaîne 'YYYY-MM-DD' en Timestamp localisé."""
    return pd.Timestamp(date_str.replace("-", ""), tz=timezone)


def resample_series(series: pd.Series | pd.DataFrame, step: str) -> pd.Series | pd.DataFrame:
    """Rééchantillonne selon le pas de temps choisi."""
    if step == "D":
        return series.resample("D").mean()
    return series  # "H" → données brutes


def step_label(step: str) -> str:
    return "journalier" if step == "D" else "horaire"


def safe_fetch(func, *args, retries: int = 3, wait: float = 5.0, **kwargs):
    """Appel API avec retries automatiques en cas d'erreur réseau."""
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt == retries:
                raise
            log.warning("Tentative %d/%d échouée (%s). Nouvelle tentative dans %.0fs…",
                        attempt, retries, exc, wait)
            time.sleep(wait)