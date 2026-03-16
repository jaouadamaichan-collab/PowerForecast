"""
=============================================================================
  Prix du gaz naturel européen (TTF) – Pas horaire – 2017 à fin 2025
=============================================================================

SOURCE PRINCIPALE : Yahoo Finance via yfinance (ticker TTF=F)
  - Données HORAIRES  : disponibles sur environ ~730 jours glissants
  - Données JOURNALIÈRES : disponibles depuis 2007 sans limite

STRATÉGIE ADOPTÉE :
  1. Télécharger TOUTES les données journalières 2017–2025  (OHLCV)
  2. Télécharger les données HORAIRES sur la fenêtre disponible (~2 ans)
  3. Pour les périodes hors portée horaire → interpolation temporelle
     depuis les données journalières (spline cubique) pour générer un
     signal horaire synthétique cohérent avec les prix réels.
  4. Fusionner les deux et exporter en CSV.

NOTE : Le TTF (Title Transfer Facility, Pays-Bas) est le benchmark
       de référence du marché gazier européen.
       Unité : EUR/MWh  (ou USD/MMBtu selon la source Yahoo)

DÉPENDANCES :
    pip install yfinance pandas numpy scipy matplotlib

=============================================================================
"""
!pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import time
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TICKER        = "TTF=F"          # Dutch TTF Natural Gas Futures (Yahoo Finance)
START_DATE    = "2023-01-01"
END_DATE      = "2025-12-31"
OUTPUT_CSV    = "ttf_gas_hourly_2023_2025.csv"
OUTPUT_PLOT   = "ttf_gas_hourly_plot.png"

# Fenêtre max pour les données horaires Yahoo Finance (~730 jours)
HOURLY_WINDOW_DAYS = 720


# ─────────────────────────────────────────────────────────────────────────────
#  FONCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def download_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Télécharge les données journalières (sans limite dans le temps)."""
    print(f"\n[1/4] Téléchargement des données JOURNALIÈRES ({start} → {end})...")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=True,
    )
    if df.empty:
        raise ValueError(f"Aucune donnée journalière trouvée pour {ticker}.")

    # Aplatir le MultiIndex si présent
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    print(f"    ✓ {len(df)} jours récupérés  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def download_hourly_chunks(ticker: str, end: str, window_days: int = 720) -> pd.DataFrame:
    """
    Télécharge les données horaires par tranches de 59 jours
    (limite Yahoo Finance par requête pour interval='1h').
    """
    print(f"\n[2/4] Téléchargement des données HORAIRES (fenêtre ~{window_days} jours)...")

    end_dt    = pd.Timestamp(end)
    start_dt  = end_dt - timedelta(days=window_days)

    chunk_size = timedelta(days=59)   # limite Yahoo pour interval=1h
    all_chunks = []

    current = start_dt
    while current < end_dt:
        chunk_end = min(current + chunk_size, end_dt)
        try:
            chunk = yf.download(
                ticker,
                start=current.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval="1h",
                auto_adjust=True,
                progress=False,
            )
            if not chunk.empty:
                if isinstance(chunk.columns, pd.MultiIndex):
                    chunk.columns = chunk.columns.get_level_values(0)
                all_chunks.append(chunk)
                print(f"    ✓ {current.date()} → {chunk_end.date()} : {len(chunk)} lignes")
            else:
                print(f"    ⚠  {current.date()} → {chunk_end.date()} : vide")
        except Exception as e:
            print(f"    ✗ Erreur {current.date()} → {chunk_end.date()} : {e}")

        current = chunk_end
        time.sleep(0.5)   # pause pour éviter le rate-limiting

    if not all_chunks:
        print("    ⚠ Aucune donnée horaire récupérée.")
        return pd.DataFrame()

    df_hourly = pd.concat(all_chunks)
    df_hourly.index = pd.to_datetime(df_hourly.index).tz_localize(None)
    df_hourly = df_hourly[~df_hourly.index.duplicated(keep="first")].sort_index()

    print(f"    ✓ Total : {len(df_hourly)} heures  "
          f"({df_hourly.index[0].date()} → {df_hourly.index[-1].date()})")
    return df_hourly


def interpolate_to_hourly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Interpole les données journalières vers un pas horaire via spline cubique.
    Simule une variation intra-journalière réaliste (cloche gaussienne).
    """
    print("\n[3/4] Interpolation horaire des données journalières...")

    # Index horaire cible
    hourly_index = pd.date_range(
        start=df_daily.index[0],
        end=df_daily.index[-1] + timedelta(hours=23),
        freq="h"
    )

    # Spline cubique sur les prix de clôture journaliers
    days_numeric  = (df_daily.index - df_daily.index[0]).days.to_numpy(dtype=float)
    close_prices  = df_daily["Close"].to_numpy(dtype=float)

    # Supprimer les NaN
    mask = ~np.isnan(close_prices)
    cs   = CubicSpline(days_numeric[mask], close_prices[mask])

    # Évaluer sur l'index horaire
    hours_numeric  = (hourly_index - df_daily.index[0]).total_seconds() / 86400.0
    prices_interp  = cs(hours_numeric)

    # Ajouter une légère variation intra-journalière (bruit réaliste)
    # Profil en cloche : légèrement plus haut en milieu de journée
    hour_of_day   = hourly_index.hour
    intraday_mult = 1.0 + 0.003 * np.sin(np.pi * (hour_of_day - 6) / 12) * (hour_of_day >= 6) * (hour_of_day <= 18)
    prices_interp = prices_interp * intraday_mult

    df_interp = pd.DataFrame(
        {"Close": prices_interp, "Source": "interpolated"},
        index=hourly_index
    )

    # Clipper les valeurs négatives éventuelles (ex: spline débordement)
    df_interp["Close"] = df_interp["Close"].clip(lower=0.1)

    print(f"    ✓ {len(df_interp)} points horaires générés par interpolation")
    return df_interp


def merge_hourly_data(df_daily: pd.DataFrame,
                      df_hourly: pd.DataFrame,
                      start: str, end: str) -> pd.DataFrame:
    """
    Fusionne les données réelles (horaires) avec l'interpolation (journalière).
    Les données réelles prennent la priorité.
    """
    print("\n[4/4] Fusion des données réelles et interpolées...")

    # Créer l'index horaire complet
    full_index = pd.date_range(start=start, end=end + " 23:00:00", freq="h")

    # Interpolation sur toute la période journalière
    df_interp = interpolate_to_hourly(df_daily)
    df_interp = df_interp.reindex(full_index)

    # Construire le DataFrame final
    result = pd.DataFrame(index=full_index)
    result["Prix_Close"]  = df_interp["Close"]
    result["Source"]      = "interpolated"

    # Injecter les données horaires réelles là où disponibles
    if not df_hourly.empty:
        # Aligner sur l'index horaire
        df_h = df_hourly.reindex(full_index)

        real_mask = df_h["Close"].notna()
        result.loc[real_mask, "Prix_Close"] = df_h.loc[real_mask, "Close"]
        result.loc[real_mask, "Source"]     = "real_hourly"

        n_real = real_mask.sum()
        print(f"    ✓ {n_real:,} points réels horaires injectés")
        print(f"    ✓ {(~real_mask).sum():,} points interpolés conservés")

    # Nettoyage final
    result.index.name = "Datetime"
    result["Prix_Close"] = result["Prix_Close"].interpolate(method="time")
    result["Ticker"]     = TICKER
    result["Unite"]      = "USD/MMBtu (Yahoo Finance)"

    total = result["Prix_Close"].notna().sum()
    print(f"\n    ✓ DataFrame final : {len(result):,} heures  |  {total:,} valeurs valides")
    return result


def plot_results(df: pd.DataFrame, output_path: str):
    """Génère un graphique de visualisation des données."""
    print(f"\nGénération du graphique → {output_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Prix du Gaz Naturel Européen (TTF)\nDonnées Horaires 2017–2025",
                 fontsize=14, fontweight="bold", y=0.98)

    # Séparer données réelles et interpolées
    real  = df[df["Source"] == "real_hourly"]["Prix_Close"]
    interp = df[df["Source"] == "interpolated"]["Prix_Close"]

    # ── Graphique principal ──
    ax1.plot(interp.index, interp.values, color="#aec6cf", linewidth=0.4,
             label="Interpolé (journalier → horaire)", alpha=0.8)
    if not real.empty:
        ax1.plot(real.index, real.values, color="#1f4e79", linewidth=0.6,
                 label="Données réelles horaires", alpha=0.9)

    ax1.set_ylabel("Prix (USD/MMBtu)", fontsize=11)
    ax1.set_title("Prix TTF – Série horaire complète", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # Mise en évidence de la crise 2022
    ax1.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
                alpha=0.08, color="red", label="Crise énergie 2022")

    # ── Graphique de la source des données ──
    source_num = (df["Source"] == "real_hourly").astype(int)
    ax2.fill_between(df.index, source_num, step="post",
                     color="#1f4e79", alpha=0.6)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Interpolé", "Réel"], fontsize=8)
    ax2.set_ylabel("Source", fontsize=9)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Graphique sauvegardé")


def print_summary(df: pd.DataFrame):
    """Affiche un résumé statistique des données."""
    print("\n" + "="*60)
    print("  RÉSUMÉ DES DONNÉES")
    print("="*60)
    print(f"  Période          : {df.index[0]} → {df.index[-1]}")
    print(f"  Nombre de points : {len(df):,} heures")
    print(f"  Données réelles  : {(df['Source']=='real_hourly').sum():,} heures")
    print(f"  Interpolées      : {(df['Source']=='interpolated').sum():,} heures")
    print(f"\n  Prix (USD/MMBtu) :")
    print(f"    Min    : {df['Prix_Close'].min():.2f}")
    print(f"    Max    : {df['Prix_Close'].max():.2f}")
    print(f"    Moyen  : {df['Prix_Close'].mean():.2f}")
    print(f"    Médian : {df['Prix_Close'].median():.2f}")

    # Stats par année
    print(f"\n  Moyenne annuelle :")
    yearly = df.groupby(df.index.year)["Prix_Close"].mean()
    for yr, val in yearly.items():
        bar = "█" * int(val / 2)
        print(f"    {yr} : {val:6.2f}  {bar}")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
#  PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  COLLECTE DES PRIX DU GAZ EUROPÉEN (TTF) – HORAIRE")
    print(f"  Période : {START_DATE} → {END_DATE}")
    print(f"  Ticker  : {TICKER}")
    print("="*60)

    # 1. Données journalières (toute la période)
    df_daily = download_daily(TICKER, START_DATE, END_DATE)

    # 2. Données horaires réelles (fenêtre limitée)
    df_hourly = download_hourly_chunks(TICKER, END_DATE, HOURLY_WINDOW_DAYS)

    # 3 & 4. Interpolation + Fusion
    df_final = merge_hourly_data(df_daily, df_hourly, START_DATE, END_DATE)

    # Export CSV
    df_final.to_csv(OUTPUT_CSV)
    print(f"\n✅ Données exportées → {OUTPUT_CSV}")

    # Graphique
    plot_results(df_final, OUTPUT_PLOT)

    # Résumé
    print_summary(df_final)

    # Aperçu
    print("\nAperçu des premières lignes :")
    print(df_final.head(5).to_string())
    print("\nAperçu des dernières lignes :")
    print(df_final.tail(5).to_string())

    return df_final


if __name__ == "__main__":
    df = main()
