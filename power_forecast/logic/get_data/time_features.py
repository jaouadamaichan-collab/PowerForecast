import numpy as np
import pandas as pd
from power_forecast.logic.utils.others import load_df, save_df
from power_forecast.params import *
from power_forecast.logic.get_data.download_api import create_dataframe_base
import pycatch22


# df_aligned = align_start_to_column(df, column='HRV', apply=True)
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


# Below -350:  25 values
# Above 2000: 7 values
# Total to replace:   32 / 2270016 (0.0014%)
# df_clean = replace_outliers_with_interpolation(df_aligned, limit_low=LIMIT_LOW, limit_high=LIMIT_HIGH)
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


def build_feature_dataframe(
    filepath:      str,
    country_objective: str  = 'FRA',
    align_column:  str  = 'HRV',
    limit_low:     float = LIMIT_LOW,
    limit_high:    float = LIMIT_HIGH,
    window:        int  = WINDOW_CATCH22,
    step:          int  = STEP_CATCH22,
    time_interval: str  = 'hourly',
    save_name:     str  = 'df_catch24_timeseries',
    load_from_pickle: bool = False,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for electricity price data.
    Runs the following steps in order:

        1. create_dataframe_base       — load raw CSV
        2. align_start_to_column       — trim to latest country start date
        3. replace_outliers_with_interpolation — clean extreme prices
        4. add_catch24_features        — rolling catch24 feature extraction
        5. add_temporal_features       — cyclical and binary time features

    ⚠️  Step 4 (catch24) can take more than 1 minute. On first run the result
        is automatically saved as a pickle. On subsequent runs, set
        load_from_pickle=True to skip recomputation and load directly.

    Parameters
    ----------
    filepath         : path to raw CSV file
    align_column     : country column to use as reference start date (default: 'HRV')
    limit_low        : lower outlier bound in €/MWh                  (default: LIMIT_LOW)
    limit_high       : upper outlier bound in €/MWh                  (default: LIMIT_HIGH)
    window           : catch24 lookback window in days               (default: WINDOW_CATCH22)
    step             : catch24 step size in days                     (default: STEP_CATCH22)
    time_interval    : frequency of input df — 'hourly' or 'daily'   (default: 'hourly')
    save_name        : pickle filename to save/load                  (default: 'df_catch24_timeseries')
    load_from_pickle : if True, skip pipeline and load from pickle   (default: False)

    Returns
    -------
    df : fully enriched DataFrame ready for modeling
    """

    # ── Shortcut: load from pickle if already computed ────────────────────────
    if load_from_pickle:
        print("⚡ Loading from pickle — skipping pipeline...")
        return load_df(save_name)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("\n── Step 1: Load raw data ────────────────────────────────────────")
    df = create_dataframe_base(filepath)

    # ── Step 2: Align start date ──────────────────────────────────────────────
    print("\n── Step 2: Align start date ─────────────────────────────────────")
    df = align_start_to_column(df, column=align_column, apply=True)

    # ── Step 3: Remove outliers ───────────────────────────────────────────────
    print("\n── Step 3: Replace outliers ─────────────────────────────────────")
    df = replace_outliers_with_interpolation(df, limit_low=limit_low, limit_high=limit_high)

    # ── Step 4: catch24 features ──────────────────────────────────────────────
    print("\n── Step 4: catch24 features (may take > 1/2 min) ──────────────────")
    df = add_catch24_features(df, window=window, step=step, time_interval=time_interval, country=country_objective)

    # ── Step 5: Temporal features ─────────────────────────────────────────────
    print("\n── Step 5: Temporal features ────────────────────────────────────")
    df = add_temporal_features(df)
    
    # ── Step 6: Meteo features ─────────────────────────────────────────────
    #print("\n── Step 6: Meteo features ───────────────────────────────────────")
    #df = meteo_features(df, country=country_objective, start_time = df.index.min(), end_time = df.index.max())

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n── Saving to pickle ─────────────────────────────────────────────")
    save_df(df, save_name)

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
