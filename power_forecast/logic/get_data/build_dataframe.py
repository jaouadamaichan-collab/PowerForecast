import numpy as np
import pandas as pd
from power_forecast.logic.get_data.kaggle_df import create_df_from_local_csv
from power_forecast.logic.get_data.time_features import (
    replace_outliers_with_interpolation,
    add_temporal_features,
    add_public_holidays,
    add_target_horizon_features,
    add_catch24_features,
    add_lag_and_contexte_features_target,
    add_lag_and_contexte_features_frontiere,
    filter_neighbor_columns,
    add_crisis_column,
    drop_boundary_nans)
from power_forecast.logic.get_data.meteo_features import get_meteo
from power_forecast.logic.get_data.entsoe_features import get_gen_load_forecast
from power_forecast.params import *
from power_forecast.logic.utils.others import load_df, save_df

def build_feature_dataframe(
    filepath:      str,
    load_from_pickle: bool = False,
    country_objective: str  = 'France',
    target_day_distance: int = 2,
    time_interval: str  = 'h',# 'h' for hourly, 'D' for daily
    save_name:     str  = 'df_with_features',
    drop_nan: bool = True,
    keep_only_neighbors: bool = True,
    add_lag_frontiere: bool = True,
    add_crisis: bool = True,
    add_gen_load_forecast: bool = True,
    add_catch24: bool = True,
) -> pd.DataFrame:


    # ── Shortcut: load from pickle if already computed ────────────────────────
    if load_from_pickle:
        print("⚡ Loading from pickle — skipping pipeline!")
        return load_df(save_name)
    
    iso_objective = VILLE_TO_ISO.get(country_objective, None)
    iso_entsoe = OBJ_ISO_TO_ENTSOE.get(iso_objective, None)
    if iso_objective is None or iso_entsoe is None:
        raise ValueError(f"Country '{country_objective}' not recognized. Available options: {VILLES_DISPONIBLES}")
    
    # ── 1. Create base dataframe from local CSV ───────────────────────────────
    df = create_df_from_local_csv(filepath)
    
    # ── 2. Align start to target country ────────────────────────────────────
    if keep_only_neighbors:
        df = filter_neighbor_columns(df, iso=iso_objective)
    
    df = drop_boundary_nans(df)

    # ── 3. Replace outliers with interpolation ───────────────────────────────
    df = replace_outliers_with_interpolation(df, limit_low=LIMIT_LOW, limit_high=LIMIT_HIGH)
    
    # Define date range for meteo and entsoe data, so we can merge at the end
    date_start = df.index.min().strftime("%Y-%m-%d")
    date_end   = (df.index.max() + pd.Timedelta(days=1 + target_day_distance)).strftime("%Y-%m-%d")
    
    # ── 4. Add meteo features ─────────────────────────────────────────────
    df_meteo   = get_meteo(country_objective, date_start, date_end, time_interval)
    print(f"Meteo for {country_objective} downloaded")
    
    # ── 5. Add gen/load forecast features from ENTSOE ───────────────────────────────
    if add_gen_load_forecast:
        df_entsoe  = get_gen_load_forecast(iso_entsoe, date_start, date_end, step=time_interval)
        print(f"Generation, Load and Forecast for {country_objective} downloaded")
    
    meteo_cache_path = Path("raw_data/pickle_files/meteo_cache") / f"meteo_{country_objective}_{date_start}_{date_end}_{time_interval}.pkl"

    if meteo_cache_path.exists():
        print(f"  ✓ Meteo cache found, loading from {meteo_cache_path}")
        df_meteo = pd.read_pickle(meteo_cache_path)
    else:
        print(f"  ✗ Meteo cache not found, fetching...")
        df_meteo = get_meteo(country_objective, date_start, date_end, time_interval)
        meteo_cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_meteo.to_pickle(meteo_cache_path)  # save directly, bypass save_df entirely
        print(f"  ✓ Meteo saved to {meteo_cache_path}")
    
    # ── 6. Add crisis and holiday features ───────────────────────────────────────────────
    if add_crisis:
        df_crisis = add_crisis_column(date_start, date_end, timezone="UTC")

    df_holidays = add_public_holidays(iso_objective, date_start, date_end, timezone="UTC")

    # ── 7. Merge all features into a single dataframe ───────────────────────────────────────────────
    frames = {"meteo": df_meteo, "holidays": df_holidays}
        
    if add_gen_load_forecast:
        frames["entsoe"] = df_entsoe
    if add_crisis:
        frames["crisis"] = df_crisis

    for name, frame in frames.items():
        # normalize timezone
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
        # align to df index
        df = df.join(frame, how="left")
    
    # ── 8. Add basic temporal features  ───────────────────────────────────────────────
    df = add_temporal_features(df)
    
    
    # ── Step 7: Add lag and context features ───────────────────────────────────────
    df = add_lag_and_contexte_features_target(df, iso_objective=iso_objective)
    
    if add_lag_frontiere:
        for neighbor in FRONTIERE.get(iso_objective, []):
            df = add_lag_and_contexte_features_frontiere(df, iso_objective=neighbor)
    
    # ── 9. Add catch24 features ───────────────────────────────────────────────      
    if add_catch24:
        df = add_catch24_features(df, window=WINDOW_CATCH22, step=STEP_CATCH22, time_interval=time_interval, country=iso_objective)

    df = df.drop(columns=[
        'hour',
        'day_of_week',
        'month',
        'quarter',
        'year',
        'day_of_year'
    ])
    
    # ── 10. Drop rows with NaN values and interpolate ───────────────────────────────────────────────
    df_clean = clean_dataframe(
        df=df,
        max_rolling_back=MAX_LAG_BACK,
        target_day_distance=target_day_distance,
        nan_threshold=DROP_COLUMN_NAN_TRESHOLD,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n── Saving to pickle ─────────────────────────────────────────────")
    save_df(df_clean, save_name)
    
    return df_clean



def clean_dataframe(
    df: pd.DataFrame,
    max_rolling_back: int,
    target_day_distance: int,
    nan_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Clean the feature dataframe by:
        1. Dropping columns with more than `nan_threshold` % of NaN values
           (typically API columns with sparse data).
        2. Dropping the first `max_rolling_back` rows with NaN due to rolling lookback.
        3. Dropping the last `H` rows with NaN due to future shift on target columns.
        4. Interpolating all remaining NaN values.

    Parameters
    ----------
    df              : pd.DataFrame with DatetimeIndex
    max_rolling_back: int, number of rows to drop at the start (rolling window size)
    H               : int, number of rows to drop at the end (target_day_distance * 24)
    nan_threshold   : float, max allowed NaN ratio per column (default 0.05 = 5%)

    Returns
    -------
    df : pd.DataFrame, cleaned with zero NaN values

    ⚠️  Call this function BEFORE train/test split to avoid
    data leakage from interpolation across the split boundary.
    """
    initial_shape = df.shape
    H = target_day_distance * 24

    # ── 1. Drop columns with more than nan_threshold NaN ─────────────────
    nan_ratio = df.isna().mean()
    cols_to_drop = nan_ratio[nan_ratio > nan_threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"  1. Dropped {len(cols_to_drop)} columns with >{nan_threshold*100:.0f}% NaN: {cols_to_drop}")

    # ── 2. Drop first rows (NaN due to rolling lookback) ─────────────────
    df = df.iloc[max_rolling_back:]
    print(f"  2. Dropped first {max_rolling_back} rows (rolling lookback)")

    # ── 3. Drop last rows (NaN due to future shift on target columns) ─────
    df = df.iloc[:-H]
    print(f"  3. Dropped last {H} rows (future shift H={H})")

    # ── 4. Interpolate remaining NaN ──────────────────────────────────────
    nan_before = df.isna().sum().sum()
    df = df.interpolate(method='time')
    df = df.fillna(method='bfill')
    nan_after = df.isna().sum().sum()
    print(f"  4. Interpolated NaN: {nan_before} → {nan_after} remaining")

    # ── Sanity check ──────────────────────────────────────────────────────
    assert df.isna().sum().sum() == 0, "⚠️ Still NaN values remaining after cleaning!"
    print(f"  ✓ Done — shape: {initial_shape} → {df.shape}, zero NaN values")

    return df
