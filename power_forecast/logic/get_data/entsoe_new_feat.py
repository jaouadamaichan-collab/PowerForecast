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

VALID_STEPS = {"H", "D"}

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
# Fonctions publiques — get_data / get_full_data
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
        Index DatetimeTZ, une colonne par pays (€/MWh).

    Examples
    --------
    df = get_data("FR", "2025-01-01", "2025-06-30")
    df = get_data(["FR", "DE_LU", "ES"], "2025-01-01", "2025-03-31", step="H")
    """
    if step not in VALID_STEPS:
        raise ValueError(f"Paramètre 'step' invalide : '{step}'. Valeurs acceptées : {VALID_STEPS}")

    countries = [country] if isinstance(country, str) else list(country)
    unknown = [c for c in countries if c not in COUNTRY_LABELS]
    if unknown:
        log.warning("Code(s) pays inconnu(s) : %s — vérifiez la liste COUNTRY_LABELS.", unknown)

    client   = EntsoePandasClient(api_key=api_key)
    ts_start = parse_date(start, timezone)
    ts_end   = parse_date(end,   timezone)

    return fetch_all_countries(client, countries, ts_start, ts_end, step=step)


def get_full_data(
    countries: str | list[str],
    target: str,
    start: str,
    end: str,
    step: str = "D",
    api_key: str = DEFAULT_API_KEY,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict[str, pd.DataFrame]:
    """
    Récupère les prix day-ahead pour une liste de pays ET les données détaillées
    (production, consommation, prévisions éolien/solaire) pour un pays cible.

    Parameters
    ----------
    countries : str ou list[str]
        Liste des pays pour les prix day-ahead. Ex : ["FR", "DE_LU", "ES"]
    target : str
        Pays cible pour les données détaillées. Doit être dans `countries`.
        Ex : "FR"
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
    dict avec les clés :
        "prices"      → pd.DataFrame  prix day-ahead, une colonne par pays (€/MWh)
        "generation"  → pd.DataFrame  production par type d'énergie pour `target` (MW)
        "load"        → pd.DataFrame  consommation pour `target` (MW)
        "forecast"    → pd.DataFrame  prévisions éolien/solaire pour `target` (MW)

    Examples
    --------
    result = get_full_data(
        countries=["FR", "DE_LU", "ES"],
        target="FR",
        start="2025-01-01",
        end="2025-06-30",
        step="H",
    )
    df_prices     = result["prices"]
    df_generation = result["generation"]
    df_load       = result["load"]
    df_forecast   = result["forecast"]
    """
    if step not in VALID_STEPS:
        raise ValueError(f"Paramètre 'step' invalide : '{step}'. Valeurs acceptées : {VALID_STEPS}")

    country_list = [countries] if isinstance(countries, str) else list(countries)

    if target not in country_list:
        log.warning(
            "Le pays cible '%s' ne fait pas partie de la liste countries (%s). "
            "Il sera ajouté automatiquement.",
            target, country_list,
        )
        country_list = [target] + country_list

    unknown = [c for c in country_list if c not in COUNTRY_LABELS]
    if unknown:
        log.warning("Code(s) pays inconnu(s) : %s — vérifiez la liste COUNTRY_LABELS.", unknown)

    client   = EntsoePandasClient(api_key=api_key)
    ts_start = parse_date(start, timezone)
    ts_end   = parse_date(end,   timezone)

    log.info("═" * 60)
    log.info("Prix day-ahead       : %s", ", ".join(country_list))
    log.info("Données détaillées   : %s", COUNTRY_LABELS.get(target, target))
    log.info("Période              : %s → %s | Pas : %s",
             ts_start.date(), ts_end.date(), step_label(step))
    log.info("═" * 60)

    # Prix day-ahead pour tous les pays
    df_prices = fetch_all_countries(client, country_list, ts_start, ts_end, step=step)

    # Données détaillées uniquement pour le pays cible
    df_generation  = fetch_actual_generation(client, target, ts_start, ts_end, step=step)
    df_load_series = fetch_actual_load(client, target, ts_start, ts_end, step=step)
    df_forecast    = fetch_wind_solar_forecast(client, target, ts_start, ts_end, step=step)

    # Homogénéisation en DataFrame
    df_load = (
        df_load_series.to_frame(name=target)
        if isinstance(df_load_series, pd.Series)
        else df_load_series
    )

    return {
        "prices":     df_prices,
        "generation": df_generation,
        "load":       df_load,
        "forecast":   df_forecast,
    }


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
    if step not in VALID_STEPS:
        raise ValueError(f"Paramètre 'step' invalide : '{step}'. Valeurs acceptées : {VALID_STEPS}")

    log.info("⬇  Récupération des prix pour %s (%s → %s)…",
             COUNTRY_LABELS.get(country, country), start.date(), end.date())

    raw    = safe_fetch(client.query_day_ahead_prices, country, start=start, end=end)
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
    Les colonnes sont les codes pays.
    """
    frames = {}
    errors = {}

    for country in countries:
        try:
            frames[country] = fetch_prices(client, country, start, end, step)
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
# Données complémentaires (pays cible uniquement)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Statistiques
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les statistiques descriptives par colonne."""
    stats = df.describe().T
    stats.index = [COUNTRY_LABELS.get(c, c) for c in stats.index]
    stats.index.name = "Pays"
    return stats.round(2)


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_prices(df: pd.DataFrame, step: str, output_dir: Path | None = None) -> None:
    """Génère le graphique des prix (multi-pays si nécessaire)."""
    n_countries = len(df.columns)
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, country in enumerate(df.columns):
        color = PALETTE[i % len(PALETTE)]
        label = COUNTRY_LABELS.get(country, country)
        ax.plot(df.index, df[country], linewidth=1.5, label=label, color=color, alpha=0.85)

    ax.set_title(
        f"Prix day-ahead — {', '.join(COUNTRY_LABELS.get(c, c) for c in df.columns)}\n"
        f"({df.index[0].strftime('%d/%m/%Y')} → {df.index[-1].strftime('%d/%m/%Y')}"
        f" | pas : {step_label(step)})",
        fontsize=13, pad=12,
    )
    ax.set_ylabel("€/MWh", fontsize=11)
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=8))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.85)

    if n_countries == 1:
        country = df.columns[0]
        ax.fill_between(df.index, df[country], alpha=0.08, color=PALETTE[0])

    plt.tight_layout()

    if output_dir:
        path = output_dir / "prix_day_ahead.png"
        fig.savefig(path, dpi=150)
        log.info("📊 Graphique sauvegardé : %s", path)

    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════

def export_results(df: pd.DataFrame, output_dir: Path, prefix: str = "prix_day_ahead") -> None:
    """Exporte les données en CSV et les statistiques en CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = output_dir / f"{prefix}.csv"
    df_export = df.copy()
    df_export.columns = [COUNTRY_LABELS.get(c, c) for c in df_export.columns]
    df_export.to_csv(csv_path, float_format="%.2f")
    log.info("💾 Données exportées : %s (%d lignes)", csv_path, len(df_export))

    stats      = compute_stats(df)
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
        help="Clé API ENTSO-E (défaut : clé configurée)",
    )
    p.add_argument(
        "--countries", nargs="+", default=DEFAULT_COUNTRIES, metavar="CODE",
        help=f"Codes pays pour les prix day-ahead (défaut : {DEFAULT_COUNTRIES}). Ex : FR DE_LU ES",
    )
    p.add_argument(
        "--target", default=DEFAULT_TARGET_COUNTRY, metavar="CODE",
        help=(
            f"Pays cible pour les données détaillées — production, consommation, "
            f"prévisions éolien/solaire (défaut : {DEFAULT_TARGET_COUNTRY}). "
            f"Doit faire partie de --countries."
        ),
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
        "--step", choices=list(VALID_STEPS), default=DEFAULT_STEP,
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
    p.add_argument("--no-export", action="store_true", help="Ne pas exporter les CSV")
    p.add_argument("--no-plot",   action="store_true", help="Ne pas afficher le graphique")
    p.add_argument("--list-countries", action="store_true",
                   help="Afficher les codes pays disponibles et quitter")
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
    args   = parser.parse_args(argv)

    if args.list_countries:
        list_countries()
        return

    api_key = args.api_key or os.environ.get("ENTSOE_API_KEY", "")
    if not api_key:
        log.error("Clé API manquante. Utilisez --api-key ou la variable ENTSOE_API_KEY.")
        sys.exit(1)

    # Garantir que le pays cible est dans la liste
    countries = list(args.countries)
    if args.target not in countries:
        log.warning(
            "Le pays cible '%s' ne fait pas partie de --countries. Ajout automatique.",
            args.target,
        )
        countries = [args.target] + countries

    result = get_full_data(
        countries = countries,
        target    = args.target,
        start     = args.start,
        end       = args.end,
        step      = args.step,
        api_key   = api_key,
        timezone  = args.timezone,
    )

    df_prices     = result["prices"]
    df_generation = result["generation"]
    df_load       = result["load"]
    df_forecast   = result["forecast"]

    target_label = COUNTRY_LABELS.get(args.target, args.target)

    # ── Affichage console ──────────────────────────────────────────────────
    print("\n📊  Prix day-ahead (tous pays) :\n")
    print(df_prices.head(10).to_string())
    print("\n📈  Statistiques des prix day-ahead :\n")
    print(compute_stats(df_prices).to_string())

    if not df_generation.empty:
        print(f"\n⚡  Production — {target_label} :\n")
        print(df_generation.head(10).to_string())
        print(f"\n📈  Statistiques production — {target_label} :\n")
        print(compute_stats(df_generation).to_string())

    if not df_load.empty:
        print(f"\n🔌  Consommation — {target_label} :\n")
        print(df_load.head(10).to_string())
        print(f"\n📈  Statistiques consommation — {target_label} :\n")
        print(compute_stats(df_load).to_string())

    if not df_forecast.empty:
        print(f"\n🌬️  Prévisions éolien/solaire — {target_label} :\n")
        print(df_forecast.head(10).to_string())
        print(f"\n📈  Statistiques prévisions — {target_label} :\n")
        print(compute_stats(df_forecast).to_string())

    # ── Export ─────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if not args.no_export:
        export_results(df_prices, output_dir, prefix="prix_day_ahead")
        if not df_generation.empty:
            export_results(df_generation, output_dir, prefix=f"production_{args.target}")
        if not df_load.empty:
            export_results(df_load, output_dir, prefix=f"consommation_{args.target}")
        if not df_forecast.empty:
            export_results(df_forecast, output_dir, prefix=f"forecast_{args.target}")

    # ── Graphique ──────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_prices(df_prices, step=args.step,
                    output_dir=output_dir if not args.no_export else None)

    log.info("✅  Terminé.")


if __name__ == "__main__":
    main()
