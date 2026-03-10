# ============================================================
# ENTSO-E API — Guide d'utilisation complet
# Bibliothèque : entsoe-py
# Installation : pip install entsoe-py pandas matplotlib
# ============================================================

!pip install entsoe-py
import pandas as pd
import matplotlib.pyplot as plt
from entsoe import EntsoePandasClient


# ── 1. CONFIGURATION ─────────────────────────────────────────
API_KEY = "82e51d13-ad8c-44f1-99bd-a000cbcc2ae3"  # Obtenir sur transparency.entsoe.eu
client = EntsoePandasClient(api_key=API_KEY)

# Définir la plage temporelle (toujours avec timezone !)
start = pd.Timestamp("20250101", tz="Europe/Paris")
end   = pd.Timestamp("20250630", tz="Europe/Paris")

country = "DE_LU"  # France — autres : 'DE_LU', 'ES', 'BE', 'IT', 'NL'...


# ── 2. PRIX DAY-AHEAD (€/MWh) ────────────────────────────────
print("📈 Prix day-ahead France (€/MWh)")
prices = client.query_day_ahead_prices(country, start=start, end=end)
print(prices.head())

prices.plot(title="Prix day-ahead — Allemagne (2015 -2025)", ylabel="€/MWh")
plt.tight_layout()
plt.show()


# ── 3. CONSOMMATION RÉELLE ────────────────────────────────────
print("\n⚡ Consommation réelle (MW)")
load = client.query_load(country, start=start, end=end)
print(load.head())


# ── 4. PRÉVISION DE CONSOMMATION ─────────────────────────────
print("\n📊 Prévision de consommation (MW)")
load_forecast = client.query_load_forecast(country, start=start, end=end)
print(load_forecast.head())


# ── 5. PRODUCTION PAR FILIÈRE ────────────────────────────────
# print("\n🏭 Production par filière (MW)")
# generation = client.query_generation(country, start=start, end=end)
# print(generation.head())

# generation.plot(
#     title="Mix de production — Allemagne (2015 -2025)",
#     ylabel="MW",
#     figsize=(14, 6)
# )
# plt.tight_layout()
# plt.show()


# ── 6. PRÉVISION ÉOLIEN + SOLAIRE ────────────────────────────
print("\n🌬️ Prévision éolien + solaire (MW)")
wind_solar = client.query_wind_and_solar_forecast(
    country, start=start, end=end, psr_type=None  # None = toutes les filières
)
print(wind_solar.head())


# ── 7. CAPACITÉ INSTALLÉE ────────────────────────────────────
print("\n🔋 Capacité installée par filière (MW)")
capacity = client.query_installed_generation_capacity(country, start=start, end=end)
print(capacity)


# ── 8. FLUX TRANSFRONTALIERS ─────────────────────────────────
print("\n🔁 Flux Allemagne → France (MW)")
cross_border = client.query_crossborder_flows("DE_LU", "FR", start=start, end=end)
print(cross_border.head())


# ── 9. CAPACITÉ D'ÉCHANGE (NTC) ──────────────────────────────
# print("\n🔌 Capacité nette de transfert DE_LU → FR (MW)")
# ntc = client.query_net_transfer_capacity_dayahead("DE_LU", "FR", start=start, end=end)
# print(ntc.head())


# ── 10. EXPORT CSV ───────────────────────────────────────────
# prices.to_csv("prix_day_ahead_fr.csv")
# # generation.to_csv("production_fr.csv")
# print("\n✅ Fichiers exportés : prix_day_ahead_fr.csv, production_fr.csv")

# ── PAS DE TEMPS ─────────────────────────────────────────────
# Choisir le pas de temps : "H" (heure) ou "D" (jour)
TIME_STEP = "D"  # ← Modifier ici : "H" pour horaire, "D" pour journalier

def resample(series, step=TIME_STEP):
    """Rééchantillonne une Series ou DataFrame selon le pas de temps choisi."""
    if step == "D":
        if isinstance(series, pd.DataFrame):
            return series.resample("D").mean()
        return series.resample("D").mean()
    else:  # "H" — pas de temps horaire (données brutes)
        return series

label_step = "journalier" if TIME_STEP == "D" else "horaire"
print(f"⏱️  Pas de temps sélectionné : {label_step} ({TIME_STEP})\n")

# ── 2. PRIX DAY-AHEAD (€/MWh) ────────────────────────────────
print("📈 Prix day-ahead (€/MWh)")
prices = resample(prices)
print(prices.head())
prices.plot(title=f"Prix day-ahead — (2015-2025) [{label_step}]", ylabel="€/MWh")
plt.tight_layout()
plt.show()
