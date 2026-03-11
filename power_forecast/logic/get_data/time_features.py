import numpy as np
import pandas as pd
from power_forecast.params import *

import pycatch22
import holidays



def align_start_to_column(df, column, apply=True):
    """
    Make dataset start from the first valid (non-NaN) value of a specific column.
    If Croatia: 2017-10-01 00:00:00
    """
    if apply:
        first_valid = df[column].first_valid_index()
        df_aligned = df[df.index >= first_valid]
        return df_aligned
    return df


def replace_outliers_with_interpolation(df, limit_low=LIMIT_LOW, limit_high=LIMIT_HIGH):
    """
    Replace values outside [limit_low, limit_high] with NaN,
    then interpolate all NaNs
    """

    # Replace outliers with NaN
    df_clean = df.copy()
    df_clean[df_clean < limit_low] = float("nan")
    df_clean[df_clean > limit_high] = float("nan")

    # Interpolate ALL NaNs (pre-existing + new outliers)
    df_interpolated = df_clean.interpolate(method="time")

    return df_interpolated


def add_catch24_features(df, window=WINDOW_CATCH22, step=STEP_CATCH22, time_interval="hourly", country='FRA'):
    """
    Extracts catch24 features using a rolling window and merges them back
    into the original dataframe, broadcasting daily features to all
    respective rows of that day.

    Parameters
    ----------
    df            : DatetimeIndex dataframe with one column per country
    window        : lookback window in days (e.g. 7 = 1 week)
    step          : step size in days (e.g. 1 = daily rows in output)
    time_interval : 'hourly' (default) or 'daily' — frequency of the input df
    country       : ISO code of a single country to compute features for (e.g. 'DEU').
                    If None, computes features for all countries.

    Returns
    -------
    df_enriched   : original df with catch24 feature columns appended
    """

    # ── Time interval multiplier ──────────────────────────────────────────────
    interval_map = {
        "hourly": 24,
        "daily": 1,
    }
    assert time_interval in interval_map, f"time_interval must be one of {list(interval_map.keys())}"

    multiplier   = interval_map[time_interval]
    window_steps = window * multiplier
    step_steps   = step   * multiplier

    # ── Country selection ─────────────────────────────────────────────────────
    if country is not None:
        assert country in df.columns, f"Country '{country}' not found in df columns"
        countries = [country]
    else:
        countries = df.columns.tolist()

    # ── Get feature names upfront ─────────────────────────────────────────────
    _test         = pycatch22.catch22_all(df[countries[0]].iloc[0:window_steps].tolist(), catch24=True, short_names=True)
    feature_names = _test["names"]

    # ── Rolling extraction ────────────────────────────────────────────────────
    records = []
    for i in range(window_steps, len(df), step_steps):
        row = {"timestamp": df.index[i]}

        for c in countries:
            w = df[c].iloc[i - window_steps:i]

            if w.isna().any():
                for feat in feature_names:
                    row[f"{c}_{feat}"] = float("nan")
                continue

            result = pycatch22.catch22_all(w.tolist(), catch24=True, short_names=True)
            for feat_name, feat_val in zip(result["names"], result["values"]):
                row[f"{c}_{feat_name}"] = feat_val

        records.append(row)

    features_df = pd.DataFrame(records).set_index("timestamp")

    # ── Merge back to original df ─────────────────────────────────────────────
    df_enriched              = df.copy()
    df_enriched["_date"]     = df_enriched.index.normalize()
    features_df.index        = features_df.index.normalize()
    features_df.index.name   = "_date"

    df_enriched = df_enriched.merge(features_df, left_on="_date", right_index=True, how="left")
    df_enriched = df_enriched.drop(columns=["_date"])

    return df_enriched


def add_temporal_features(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index  # DatetimeIndex

    # Temporal components
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["year"] = idx.year
    df["day_of_year"] = idx.dayofyear

    # Binary flags
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_monday"] = (idx.dayofweek == 0).astype(int)  # typically high demand

    # Cyclical encoding (sin/cos)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365)

    # Electricity-specific flags
    df["is_peak_hour"] = idx.hour.isin(range(8, 20)).astype(int)  # 08:00-20:00
    df["is_offpeak_hour"] = idx.hour.isin(range(0, 8)).astype(int)  # 00:00-08:00
    df["is_summer"] = idx.month.isin([6, 7, 8]).astype(int)
    df["is_winter"] = idx.month.isin([12, 1, 2]).astype(int)

    return df


def add_public_holidays(df, country):
    """
    Add a binary is_holiday column for a given country based on public holidays.
    If the country is not found in COUNTRY_HOLIDAY_MAP, the column is filled with 0.

    Parameters
    ----------
    df      : DataFrame with DatetimeIndex
    country : ISO column name (e.g. 'DEU')

    Returns
    -------
    df : original df with one new binary column 'is_holiday_{country}'
    """
    df = df.copy()

    if country not in COUNTRY_HOLIDAY_MAP:
        print(f"⚠️  '{country}' not in COUNTRY_HOLIDAY_MAP — filling is_holiday_{country} with 0")
        df[f"is_holiday_{country}"] = 0
        return df

    holiday_code = COUNTRY_HOLIDAY_MAP[country]
    years        = df.index.year.unique().tolist()
    country_hols = holidays.country_holidays(holiday_code, years=years)

    # use .date() to get datetime.date objects matching the holidays library format
    df[f"is_holiday_{country}"] = pd.Series(df.index.date, index=df.index).isin(country_hols).astype(int)

    return df


def add_target_horizon_features(df: pd.DataFrame, iso_objective: str, target_day_distance: int):
    """
    df: your full timeseries dataframe (hourly)
    iso_objective: ISO code for the country
    target_day_distance: e.g. 2 for predicting 48h ahead
    """
    H = target_day_distance * 24  # shift in hours

    # --- Target temporal features (from existing columns) ---
    temporal_cols = [
        "hour", "day_of_week", "month", "quarter",
        "is_weekend", "is_monday", "is_holiday_FRA",
        "hour_sin", "hour_cos",
        "day_of_week_sin", "day_of_week_cos",
        "month_sin", "month_cos",
        "day_of_year_sin", "day_of_year_cos",
        "is_peak_hour", "is_offpeak_hour",
        "is_summer", "is_winter",
    ]
    for col in temporal_cols:
        df[f"target_{col}"] = df[col].shift(-H)

    # --- Meteo forecast (proxy: actual meteo at target time) ---
    meteo_cols = [
        f"{iso_objective}_Température (°C)",
        f"{iso_objective}_Précipitations (mm)",
        f"{iso_objective}_Vent (km/h)",
        f"{iso_objective}_Rafales (km/h)",
        f"{iso_objective}_Ensoleillement (MJ/m²)",
    ]
    for col in meteo_cols:
        df[f"target_{col}"] = df[col].shift(-H)

    # --- Price lag anchors (safe: looking into the past) ---
    # These are NOT shifted forward — they look backwards from current t
    df[f"{iso_objective}_price_lag_1day"]   = df[iso_objective].shift(24)
    df[f"{iso_objective}_price_lag_2day"]   = df[iso_objective].shift(48)
    df[f"{iso_objective}_price_lag_3day"]   = df[iso_objective].shift(72)
    df[f"{iso_objective}_price_lag_4day"]   = df[iso_objective].shift(96)
    df[f"{iso_objective}_price_lag_1week"]  = df[iso_objective].shift(7*24)

    return df

















# FEATURES add_catch24_features for each country on window and step:
# features_short = [
#     'mode_5',               # Most common price range (coarse) — where prices "sit" most often
#     'mode_10',              # Most common price range (fine) — more precise price clustering
#     'acf_timescale',        # How fast price autocorrelation decays — short=noisy, long=persistent
#     'acf_first_min',        # Lag of first autocorrelation minimum — typical price cycle length
#     'ami2',                 # Non-linear dependency between lagged values — hidden price patterns
#     'trev',                 # Time reversibility — is the price series asymmetric (spikes vs drops)?
#     'high_fluctuation',     # Proportion of large hour-to-hour changes — overall volatility
#     'stretch_high',         # Longest consecutive run above mean — sustained high price periods
#     'transition_matrix',    # Self-similarity across time — how repetitive is the price behavior?
#     'periodicity',          # Dominant cycle length — detects 24h or 168h seasonality
#     'embedding_dist',       # Structure in 2D phase space — complexity of price dynamics
#     'ami_timescale',        # Lag where mutual information first drops — non-linear memory length
#     'whiten_timescale',     # Timescale after removing linear structure — residual complexity
#     'outlier_timing_pos',   # Timing of positive price spikes — when do peaks tend to occur?
#     'outlier_timing_neg',   # Timing of negative price spikes — when do price crashes occur?
#     'low_freq_power',       # Energy in low frequencies — strength of slow trend component
#     'stretch_decreasing',   # Longest consecutive price decrease — bearish pressure duration
#     'entropy_pairs',        # Entropy of consecutive value pairs — randomness/predictability
#     'rs_range',             # Rescaled range scaling — Hurst-like long memory indicator
#     'dfa',                  # Detrended fluctuation analysis — long-range dependence strength
#     'centroid_freq',        # Center of mass of power spectrum — dominant frequency of prices
#     'forecast_error',       # Error of naive local mean forecast — how predictable is the series?
#      'mean',                # Average price level — overall costliness of electricity
#      'std'                  # Price variability — how much do prices fluctuate around the mean?
# ]

# # Your setup:
# window_size = 168  # catch22 sees 1 full week → captures weekly patterns ✅
# step        = 24   # one feature row per day  → ~365 rows for 2024 ✅

# # If you changed step=1:
# # → same feature quality, but ~8,700 rows (one per hour) and 29x slower
