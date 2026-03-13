"""
╔══════════════════════════════════════════════════════════════╗
║         ENTSO-E — Récupération automatisée des prix          ║
║         Day-Ahead pour un ou plusieurs pays                  ║
╚══════════════════════════════════════════════════════════════╝

Usage :
    python entsoe_prices_fetcher.py
    python entsoe_prices_fetcher.py --countries FR DE_LU ES --start 2025-01-01 --end 2025-06-30
    python entsoe_prices_fetcher.py --countries FR --step H --no-plot
    python entsoe_prices_fetcher.py --list-countries

Installation :
    pip install entsoe-py pandas matplotlib
"""

import argparse
import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from power_forecast.params import *

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from power_forecast.logic.utils.graphs import step_label, plot_prices

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


# ══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Fonction principale — get_data
# ══════════════════════════════════════════════════════════════════════════════

def get_data(
    country: str | list[str],
    start: str,
    end: str,
    step: str = "D",
    api_key: str = DEFAULT_API_KEY,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """
    Récupère les prix day-ahead ENTSO-E pour un ou plusieurs pays.

    Parameters
    ----------
    country : str ou list[str]
        Code(s) pays ENTSO-E. Ex : "FR", ["FR", "DE_LU", "ES"]
        Codes disponibles : FR, DE_LU, ES, BE, NL, IT, CH, AT, PT, PL…
    start : str
        Date de début au format "YYYY-MM-DD". Ex : "2025-01-01"
    end : str
        Date de fin au format "YYYY-MM-DD". Ex : "2025-06-30"
    step : str, optional
        Pas de temps : "H" (horaire) ou "D" (journalier, défaut).
    api_key : str, optional
        Clé API ENTSO-E (transparency.entsoe.eu). Utilise la clé par défaut si non fournie.
    timezone : str, optional
        Fuseau horaire (défaut : "Europe/Paris").

    Returns
    -------
    pd.DataFrame
        Index DatetimeTZ, une colonne par pays (code ENTSO-E), valeurs en €/MWh.

    Examples
    --------
    # Un seul pays
    df = get_data("FR", "2025-01-01", "2025-06-30")

    # Plusieurs pays, pas horaire
    df = get_data(["FR", "DE_LU", "ES"], "2025-01-01", "2025-03-31", step="H")

    # Clé API personnalisée
    df = get_data("BE", "2024-01-01", "2024-12-31", api_key="votre_clé_ici")
    """
    # Normalisation : un seul pays → liste
    countries = [country] if isinstance(country, str) else list(country)

    # Validation des codes pays
    unknown = [c for c in countries if c not in COUNTRY_LABELS]
    if unknown:
        log.warning("Code(s) pays inconnu(s) : %s — vérifiez la liste COUNTRY_LABELS.", unknown)

    # Initialisation client
    client = EntsoePandasClient(api_key=api_key)

    # Conversion des dates
    ts_start = parse_date(start, timezone)
    ts_end   = parse_date(end,   timezone)

    # Récupération
    return fetch_all_countries(client, countries, ts_start, ts_end, step=step)


# ══════════════════════════════════════════════════════════════════════════════
# Récupération des prix pour un pays
# ══════════════════════════════════════════════════════════════════════════════

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
    log.info("⬇  Récupération des prix pour %s (%s → %s)…",
             COUNTRY_LABELS.get(country, country), start.date(), end.date())

    raw = safe_fetch(client.query_day_ahead_prices, country, start=start, end=end)
    prices = resample_series(raw, step)

    log.info("   ✓ %d points récupérés (pas : %s)", len(prices), step_label(step))
    return prices


# ══════════════════════════════════════════════════════════════════════════════
# Récupération multi-pays
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_countries(
    client: EntsoePandasClient,
    countries: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    step: str = DEFAULT_STEP,
) -> pd.DataFrame:
    """
    Récupère les prix day-ahead pour plusieurs pays et renvoie un DataFrame.
    Les colonnes sont les codes pays (+ label en commentaire).
    """
    frames = {}
    errors = {}

    for country in countries:
        try:
            series = fetch_prices(client, country, start, end, step)
            frames[country] = series
        except Exception as exc:
            log.error("❌  Impossible de récupérer les données pour %s : %s", country, exc)
            errors[country] = str(exc)

    if not frames:
        raise RuntimeError("Aucune donnée récupérée. Vérifiez la clé API et les codes pays.")

    df = pd.DataFrame(frames)
    df.index.name = "datetime"

    if errors:
        log.warning("Pays en erreur : %s", ", ".join(errors.keys()))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Statistiques
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les statistiques descriptives par pays."""
    stats = df.describe().T  # min, max, mean, std, percentiles
    stats.index = [COUNTRY_LABELS.get(c, c) for c in stats.index]
    stats.index.name = "Pays"
    return stats.round(2)



# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════

def export_results(df: pd.DataFrame, output_dir: Path, prefix: str = "prix_day_ahead") -> None:
    """Exporte les données en CSV et les statistiques en CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV principal
    csv_path = output_dir / f"{prefix}.csv"
    df_export = df.copy()
    df_export.columns = [COUNTRY_LABELS.get(c, c) for c in df_export.columns]
    df_export.to_csv(csv_path, float_format="%.2f")
    log.info("💾 Données exportées : %s (%d lignes)", csv_path, len(df_export))

    # Statistiques
    stats = compute_stats(df)
    stats_path = output_dir / f"{prefix}_stats.csv"
    stats.to_csv(stats_path)
    log.info("📋 Statistiques exportées : %s", stats_path)


# ══════════════════════════════════════════════════════════════════════════════
# Interface en ligne de commande
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Récupération automatisée des prix day-ahead ENTSO-E",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--api-key", default=DEFAULT_API_KEY,
        help=f"Clé API ENTSO-E (défaut : clé configurée)",
    )
    p.add_argument(
        "--countries", nargs="+", default=DEFAULT_COUNTRIES, metavar="CODE",
        help=f"Codes pays ENTSO-E (défaut : {DEFAULT_COUNTRIES}). Ex : FR DE_LU ES",
    )
    p.add_argument(
        "--start", default=DEFAULT_START, metavar="YYYY-MM-DD",
        help=f"Date de début (défaut : {DEFAULT_START})",
    )
    p.add_argument(
        "--end", default=DEFAULT_END, metavar="YYYY-MM-DD",
        help=f"Date de fin (défaut : {DEFAULT_END})",
    )
    p.add_argument(
        "--step", choices=["H", "D"], default=DEFAULT_STEP,
        help="Pas de temps : H=horaire, D=journalier (défaut : D)",
    )
    p.add_argument(
        "--timezone", default=DEFAULT_TIMEZONE,
        help=f"Fuseau horaire (défaut : {DEFAULT_TIMEZONE})",
    )
    p.add_argument(
        "--output-dir", default=".", metavar="DIR",
        help="Répertoire de sortie pour les exports (défaut : répertoire courant)",
    )
    p.add_argument(
        "--no-export", action="store_true",
        help="Ne pas exporter les CSV",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Ne pas afficher le graphique",
    )
    p.add_argument(
        "--list-countries", action="store_true",
        help="Afficher les codes pays disponibles et quitter",
    )
    return p


def list_countries() -> None:
    print("\n📋  Zones disponibles (code ENTSO-E → pays)\n")
    for code, label in COUNTRY_LABELS.items():
        print(f"   {code:<10}  {label}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_countries:
        list_countries()
        return

    # ── Initialisation client ──────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("ENTSOE_API_KEY", "")
    if not api_key:
        log.error("Clé API manquante. Utilisez --api-key ou la variable ENTSOE_API_KEY.")
        sys.exit(1)

    client = EntsoePandasClient(api_key=api_key)

    # ── Plage temporelle ───────────────────────────────────────────────────
    start = parse_date(args.start, args.timezone)
    end   = parse_date(args.end,   args.timezone)

    log.info("═" * 60)
    log.info("Pays      : %s", ", ".join(args.countries))
    log.info("Période   : %s → %s", start.date(), end.date())
    log.info("Pas       : %s", step_label(args.step))
    log.info("Fuseau    : %s", args.timezone)
    log.info("═" * 60)

    # ── Récupération ───────────────────────────────────────────────────────
    df = fetch_all_countries(client, args.countries, start, end, step=args.step)

    # ── Affichage console ──────────────────────────────────────────────────
    print("\n📊  Aperçu des données :\n")
    print(df.head(10).to_string())

    print("\n📈  Statistiques :\n")
    print(compute_stats(df).to_string())

    # ── Export ─────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if not args.no_export:
        export_results(df, output_dir)

    # ── Graphique ──────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_prices(df, step=args.step, output_dir=output_dir if not args.no_export else None)

    log.info("✅  Terminé.")


if __name__ == "__main__":
    main()



