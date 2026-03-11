import numpy as np
import pandas as pd
from power_forecast.logic.get_data.time_features import (
    align_start_to_column,
    replace_outliers_with_interpolation,
    add_temporal_features,
    add_public_holidays,
    add_target_horizon_features,
    add_catch24_features,
    add_lag_and_contexte_features)
from power_forecast.logic.utils.others import load_df, save_df
from power_forecast.logic.get_data.meteo import get_meteo
from power_forecast.params import *


# A utiliser avec le raw_data all_countries.csv
def create_dataframe_base(filepath: str) -> pd.DataFrame:
    """
    Transforms raw electricity price CSV into a time series DataFrame.

    - Index: UTC datetime
    - Columns: one per country (ISO3 code)
    - Values: Price (EUR/MWhe)
    """
    # Load raw CSV

    df = pd.read_csv(filepath)
    # Parse UTC datetime & set as index
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"], utc=True)

    # Pivot: rows = datetime, columns = country, values = price
    df_pivot = df.pivot_table(
        index="Datetime (UTC)",
        columns="ISO3 Code",
        values="Price (EUR/MWhe)",
        fill_value=np.nan,
        aggfunc="first",
        dropna=False,
    )

    # Clean column name metadata
    df_pivot.columns.name = None
    df_pivot.index.name = "datetime_utc"

    # Sort index
    df_pivot = df_pivot.sort_index()
    df_pivot = df_pivot.drop("MKD", axis=1)

    return df_pivot


def build_feature_dataframe(
    filepath:      str,
    country_objective: str  = 'France',
    target_day_distance: int = 2,
    align_column:  str  = 'HRV',
    limit_low:     float = LIMIT_LOW,
    limit_high:    float = LIMIT_HIGH,
    window:        int  = WINDOW_CATCH22,
    step:          int  = STEP_CATCH22,
    time_interval: str  = 'hourly',
    save_name:     str  = 'df_catch24_timeseries',
    drop_nan: bool = True,
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

    iso_objective = VILLE_TO_ISO.get(country_objective, None)
    if iso_objective is None:
        raise ValueError(f"Country '{country_objective}' not found in VILLE_TO_ISO mapping. Please check the country name or update the mapping.")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("\n── Step 1: Load raw data ────────────────────────────────────────")
    df = create_dataframe_base(filepath)

    # ── Step 2: Align start date ──────────────────────────────────────────────
    print("\n── Step 2: Align start date ─────────────────────────────────────")
    df = align_start_to_column(df, column=align_column, apply=True)

    # ── Step 3: Remove outliers ───────────────────────────────────────────────
    print("\n── Step 3: Replace outliers ─────────────────────────────────────")
    df = replace_outliers_with_interpolation(df, limit_low=limit_low, limit_high=limit_high)


    # ── Step 4: Temporal features ─────────────────────────────────────────────
    print("\n── Step 4: Temporal features ────────────────────────────────────")
    df = add_temporal_features(df)
    df = add_public_holidays(df, country=iso_objective)

    # ── Step 5: Meteo features ─────────────────────────────────────────────
    print("\n── Step 5: Meteo features ───────────────────────────────────────")
    date_start = df.index.min().strftime("%Y-%m-%d")
    date_end = (df.index.max() + pd.Timedelta(days=1+target_day_distance)).strftime("%Y-%m-%d")
    df = get_meteo(df, country=country_objective, date_start=date_start, date_end=date_end)

    # ── Step 6: Add features based on target distance ───────────────────────────────────────
    print("\n── Step 6: Add features based on target distance ────────────────")
    df = add_target_horizon_features(df, iso_objective=iso_objective, target_day_distance=target_day_distance)

    # ── Step 7: Add lag and context features ───────────────────────────────────────
    print("\n── Step 7: Add lag and context feature ────────────────")
    df = add_lag_and_contexte_features(df, iso_objective=iso_objective)

    # ── Step 8: catch24 features ──────────────────────────────────────────────
    print(f"\n── Step 7: catch24 features for {country_objective} ──────────────────")
    df = add_catch24_features(df, window=window, step=step, time_interval=time_interval, country=iso_objective)

    df = df.drop(columns=[
        'hour',
        'day_of_week',
        'month',
        'quarter',
        'year',
        'day_of_year'
    ])

    if drop_nan:
        df_clean = df.dropna()
        print(f"Rows dropped: {len(df) - len(df_clean)} to avoid nan due to target distance and catch22 features")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n── Saving to pickle ─────────────────────────────────────────────")
    save_df(df_clean, save_name)

    return df_clean
