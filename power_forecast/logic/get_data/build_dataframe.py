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
    # ── 10. Drop rows with NaN values (due to target distance and catch24 features) ───────────────────────────────────────────────
    if drop_nan:
        df_clean = df.dropna()
        print(f"Rows dropped: {len(df) - len(df_clean)} to avoid nan due to target distance and catch22 features")
    
    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n── Saving to pickle ─────────────────────────────────────────────")
    save_df(df_clean, save_name)
    
    return df


