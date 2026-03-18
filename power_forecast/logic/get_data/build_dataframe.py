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
from power_forecast.logic.models.registry import load_df, save_df_topickle



def build_common_dataframe(
    filepath: str,
    country_objective: str = "France",
    target_day_distance: int = 2,
    time_interval: str = "h",
    keep_only_neighbors: bool = True,
    add_meteo: bool = True,
    add_crisis: bool = True,
    add_entsoe: bool = False,  # ATTENTION: PAS SURE D'AVOIR CA POUR LE FUTUR
) -> pd.DataFrame:
    """
    Build a common feature dataframe for electricity price forecasting.

    Runs the following steps in order:
        0. Shortcut: if load_from_pickle=True, loads and returns the previously saved DataFrame.
        1. Load base price data from a local CSV file.
        2. Optionally filter columns to keep only neighboring countries.
        3. Replace outliers with interpolated values.
        4. Fetch (or load from cache) meteorological features.
        5. Fetch (or load from cache) ENTSOE generation/load forecast features.
        6. Build crisis and public holiday features.
        7. Merge all feature DataFrames into the base DataFrame (left join on index).
        8. Drop columns with more than DROP_COLUMN_NAN_THRESHOLD % of NaN values.
        9. Deduplicate columns and rows (keep first occurrence).
        10. Interpolate remaining NaN values.
        11. Save the final DataFrame to pickle.

    Parameters
    ----------
    filepath             : str, path to the local CSV file with raw price data
    load_from_pickle     : bool, if True skip the pipeline and load from a saved pickle
    country_objective    : str, target country name (e.g. 'France')
    target_day_distance  : int, number of days ahead to forecast (used to extend date range)
    time_interval        : str, resampling frequency ('h' for hourly, 'D' for daily)
    save_name            : str, name used to save/load the pickle file
    keep_only_neighbors  : bool, if True keep only neighboring countries' columns
    add_crisis           : bool, if True add crisis period features
    add_entsoe           : bool, if True add ENTSOE generation/load forecast features

    Returns
    -------
    df : pd.DataFrame with DatetimeIndex (UTC), fully merged and cleaned features

    ⚠️  Notes:
        - Meteo and ENTSOE data are cached locally to speed up subsequent runs.
        - Set load_from_pickle=True on second run to skip the full pipeline.
        - The date range is extended by (target_day_distance + 1) days to ensure
          future shift columns (target_*) have enough data.
    """

    iso_objective = VILLE_TO_ISO.get(country_objective, None)
    iso_entsoe = OBJ_ISO_TO_ENTSOE.get(iso_objective, None)
    if iso_objective is None or iso_entsoe is None:
        raise ValueError(
            f"Country '{country_objective}' not recognized. Available options: {VILLES_DISPONIBLES}"
        )

    # ── 1. Create base dataframe from local CSV ───────────────────────────────
    df = create_df_from_local_csv(filepath)

    # ── 2. Align start to target country ─────────────────────────────────────
    if keep_only_neighbors:
        df = filter_neighbor_columns(df, iso=iso_objective)

    # ── 3. Replace outliers with interpolation ────────────────────────────────
    df = replace_outliers_with_interpolation(
        df, limit_low=LIMIT_LOW, limit_high=LIMIT_HIGH
    )

    # Define date range for meteo and entsoe data
    date_start = df.index.min().strftime("%Y-%m-%d")
    date_end = (df.index.max() + pd.Timedelta(days=1 + target_day_distance)).strftime(
        "%Y-%m-%d"
    )

    # ── 4. Add meteo features ─────────────────────────────────────────────────
    if add_meteo:
        meteo_cache_path = (
            Path("raw_data/pickle_files/meteo_cache")
            / f"meteo_{country_objective}_{date_start}_{date_end}_{time_interval}.pkl"
        )

        if meteo_cache_path.exists():
            print(f"  ✓ Meteo cache found, loading from {meteo_cache_path}")
            df_meteo = pd.read_pickle(meteo_cache_path)
        else:
            print(f"  ✗ Meteo cache not found, fetching...")
            df_meteo = get_meteo(country_objective, date_start, date_end, time_interval)
            meteo_cache_path.parent.mkdir(parents=True, exist_ok=True)
            df_meteo.to_pickle(meteo_cache_path)
            print(f"  ✓ Meteo saved to {meteo_cache_path}")

    # ── 5. Add ENTSOE generation/load forecast features ───────────────────────
    if add_entsoe:
        entsoe_cache_path = (
            Path("raw_data/pickle_files/entsoe_cache")
            / f"entsoe_{country_objective}_{date_start}_{date_end}_{time_interval}.pkl"
        )

        if entsoe_cache_path.exists():
            print(f"  ✓ Entsoe cache found, loading from {entsoe_cache_path}")
            df_entsoe = pd.read_pickle(entsoe_cache_path)
        else:
            print(f"  ✗ Entsoe cache not found, fetching...")
            df_entsoe = get_gen_load_forecast(
                iso_entsoe, date_start, date_end, step=time_interval
            )
            entsoe_cache_path.parent.mkdir(parents=True, exist_ok=True)
            df_entsoe.to_pickle(entsoe_cache_path)
            print(f"  ✓ Entsoe saved to {entsoe_cache_path}")

    # ── 6. Add crisis and public holiday features ─────────────────────────────
    if add_crisis:
        df_crisis = add_crisis_column(date_start, date_end, timezone="UTC")

    df_holidays = add_public_holidays(
        iso_objective, date_start, date_end, timezone="UTC"
    )

    # ── 7. Merge all features into a single dataframe ─────────────────────────
    frames = {"holidays": df_holidays}

    if add_meteo:
        frames["meteo"] = df_meteo
    if add_entsoe:
        frames["entsoe"] = df_entsoe
    if add_crisis:
        frames["crisis"] = df_crisis

    for name, frame in frames.items():
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
        df = df.join(frame, how="left")

    # ── 8. Drop columns with too many NaN ────────────────────────────────────
    nan_ratio = df.isna().mean()
    cols_to_drop = nan_ratio[nan_ratio > DROP_COLUMN_NAN_TRESHOLD].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(
        f"  Dropped {len(cols_to_drop)} columns with >{DROP_COLUMN_NAN_TRESHOLD*100:.0f}% NaN: {cols_to_drop}"
    )

    # ── 9. Deduplicate columns and rows ───────────────────────────────────────
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df[~df.index.duplicated(keep="first")]

    # ── 10. Interpolate remaining NaN ─────────────────────────────────────────
    nan_before = df.isna().sum().sum()
    df = df.interpolate(method="time")
    df = df.bfill()
    nan_after = df.isna().sum().sum()
    print(f"  Interpolated NaN: {nan_before} → {nan_after} remaining")

    assert df.isna().sum().sum() == 0, "⚠️ Still NaN values remaining after cleaning!"

    return df


def add_features_XGB(
    df: pd.DataFrame,
    country_objective: str,
    target_day_distance: int,
    add_lag_frontiere: bool = False,
    drop_initial_nans: bool = False,
) -> pd.DataFrame:
    """
    Add features specific for XGBoost model.

    Steps:
        1. Cyclical time encodings (hour, day of week, month, day of year)
           using sin/cos to preserve cyclical continuity.
        2. Binary electricity-specific flags (weekend, monday, peak hours, season).
        3. Lag features on the target country price column (anti-leakage shift applied).
        4. Rolling statistics on the target country price column (mean, std, min, max, range).
        5. Optionally: lag and rolling features for neighboring countries (FRONTIERE dict).
        6. Optionally: drop the first MAX_LAG_BACK_XGB rows that contain NaN
           due to the lag/rolling lookback.

    Parameters
    ----------
    df                  : pd.DataFrame with DatetimeIndex
    iso_objective       : str, ISO code of the target country (e.g. 'FR')
    target_day_distance : int, forecast horizon in days — used for anti-leakage shift
                          (shift = target_day_distance * 24 hours)
    add_lag_frontiere   : bool, if True add lag/rolling features for neighboring countries
                          defined in FRONTIERE[iso_objective]
    drop_initial_nans   : bool, if True drop the first MAX_LAG_BACK_XGB rows which
                          contain NaN due to rolling/lag lookback

    Returns
    -------
    df_with_lags : pd.DataFrame with all original columns + new feature columns
    """
    # ── Fix 1: copy and define idx FIRST ─────────────────────────────────
    df = df.copy()
    idx = df.index
    H = target_day_distance * 24
    iso_objective = VILLE_TO_ISO.get(country_objective, None)

    # ── 1. Cyclical time encodings ────────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365)

    # ── 2. Binary flags ───────────────────────────────────────────────────
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_monday"] = (idx.dayofweek == 0).astype(int)
    df["is_peak_hour"] = idx.hour.isin(range(8, 20)).astype(int)  # 08:00–20:00
    df["is_offpeak_hour"] = idx.hour.isin(range(0, 8)).astype(int)  # 00:00–08:00
    df["is_summer"] = idx.month.isin([6, 7, 8]).astype(int)
    df["is_winter"] = idx.month.isin([12, 1, 2]).astype(int)

    # ── 3 & 4. Lag and rolling features on target price ───────────────────
    target_lags_col = {}

    for lag in LAGS_XGB_TARGET:
        target_lags_col[f"{iso_objective}_lag_{lag}h"] = df[iso_objective].shift(lag)

    for w in ROLLING_WINDOWS_XGB_TARGET:
        base = df[iso_objective].shift(H)
        roll_min = base.rolling(w).min()
        roll_max = base.rolling(w).max()

        target_lags_col[
            f"{iso_objective}_roll_mean_{w}h_from_{target_day_distance}d_ago"
        ] = base.rolling(w).mean()
        target_lags_col[
            f"{iso_objective}_roll_std_{w}h_from_{target_day_distance}d_ago"
        ] = base.rolling(w).std()
        target_lags_col[
            f"{iso_objective}_roll_min_{w}h_from_{target_day_distance}d_ago"
        ] = roll_min
        target_lags_col[
            f"{iso_objective}_roll_max_{w}h_from_{target_day_distance}d_ago"
        ] = roll_max
        target_lags_col[
            f"{iso_objective}_roll_range_{w}h_from_{target_day_distance}d_ago"
        ] = (roll_max - roll_min)

    df_with_lags = pd.concat(
        [df, pd.DataFrame(target_lags_col, index=df.index)], axis=1
    )

    # ── 5. Lag and rolling features on neighboring countries ──────────────
    frontiere_list = FRONTIERE[iso_objective]
    if add_lag_frontiere:
        frontiere_lags_col = {}

        for neighbor in frontiere_list:  # ← Fix 4: iterate neighbors
            if neighbor not in df.columns:
                print(f"  ⚠️ Neighbor '{neighbor}' not found in df, skipping.")
                continue

            for lag in LAGS_XGB_FRONTIERE:
                frontiere_lags_col[f"{neighbor}_lag_{lag}h"] = df[neighbor].shift(lag)

            for w in ROLLING_WINDOWS_FRONTIERE:
                base = df[neighbor].shift(H)

                frontiere_lags_col[
                    f"{neighbor}_roll_mean_{w}h_from_{target_day_distance}d_ago"
                ] = base.rolling(w).mean()
                frontiere_lags_col[
                    f"{neighbor}_roll_std_{w}h_from_{target_day_distance}d_ago"
                ] = base.rolling(w).std()

        df_with_lags = pd.concat(
            [df_with_lags, pd.DataFrame(frontiere_lags_col, index=df_with_lags.index)],
            axis=1,
        )

    # ── 6. DROP ORIGINAL PRICE COLUMNS AND EVENTUAL initial NaN rows from lag/rolling lookback ────────────────
    df_with_lags = df_with_lags.drop(columns=frontiere_list, errors="ignore")

    if drop_initial_nans:
        df_with_lags = df_with_lags.iloc[MAX_LAG_BACK_XGB:]  # ← Fix 6: correct variable
        print(f"  Dropped first {MAX_LAG_BACK_XGB} rows (rolling lookback)")

    return df_with_lags


def add_features_RNN(
    df: pd.DataFrame,
    country_objective: str,
    target_day_distance: int,
    add_catch24: bool = False,
    add_future_time_features: bool = True,
    add_future_meteo: bool = False,
) -> pd.DataFrame:
    """
    Add features specific for XGBoost model.

    Steps:
        1. Cyclical time encodings (hour, day of week, month, day of year)
           using sin/cos to preserve cyclical continuity.
        2. Binary electricity-specific flags (weekend, monday, peak hours, season).
        3. Lag features on the target country price column (anti-leakage shift applied).
        4. Rolling statistics on the target country price column (mean, std, min, max, range).
        5. Optionally: lag and rolling features for neighboring countries (FRONTIERE dict).
        6. Optionally: drop the first MAX_LAG_BACK_XGB rows that contain NaN
           due to the lag/rolling lookback.

    Parameters
    ----------
    df                  : pd.DataFrame with DatetimeIndex
    iso_objective       : str, ISO code of the target country (e.g. 'FR')
    target_day_distance : int, forecast horizon in days — used for anti-leakage shift
                          (shift = target_day_distance * 24 hours)
    add_lag_frontiere   : bool, if True add lag/rolling features for neighboring countries
                          defined in FRONTIERE[iso_objective]
    drop_initial_nans   : bool, if True drop the first MAX_LAG_BACK_XGB rows which
                          contain NaN due to rolling/lag lookback

    Returns
    -------
    df_with_lags : pd.DataFrame with all original columns + new feature columns
    """
    # ── Fix 1: copy and define idx FIRST ─────────────────────────────────
    df = df.copy()
    idx = df.index
    H = target_day_distance * 24
    iso_objective = VILLE_TO_ISO.get(country_objective, None)

    # ── 1. Cyclical time encodings ────────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365)

    # ── 2. Binary flags ───────────────────────────────────────────────────
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_monday"] = (idx.dayofweek == 0).astype(int)
    df["is_peak_hour"] = idx.hour.isin(range(8, 20)).astype(int)  # 08:00–20:00
    df["is_offpeak_hour"] = idx.hour.isin(range(0, 8)).astype(int)  # 00:00–08:00
    df["is_summer"] = idx.month.isin([6, 7, 8]).astype(int)
    df["is_winter"] = idx.month.isin([12, 1, 2]).astype(int)

    if add_future_time_features:
        # --- Target temporal features (from existing columns) ---
        temporal_cols = [
            "is_weekend",
            "is_monday",
            f"is_holiday_{iso_objective}",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "day_of_year_sin",
            "day_of_year_cos",
            "is_peak_hour",
            "is_offpeak_hour",
            "is_summer",
            "is_winter",
        ]
        for col in temporal_cols:
            df[f"future_{col}"] = df[col].shift(-H)

    if add_future_meteo:
        # --- Meteo forecast (proxy: actual meteo at target time) ---
        meteo_cols = [
            f"{iso_objective}_temperature_c",
            f"{iso_objective}_precipitation_mm",
            f"{iso_objective}_vent_km_h",
            f"{iso_objective}_rafales_km_h",
            f"{iso_objective}_irradiation_MJ_m2",
        ]
        for col in meteo_cols:
            if col not in df.columns:
                print(f"  ⚠️ Meteo column '{col}' not found, skipping.")
                continue
            df[f"{col}_future"] = df[col].shift(-H)

    # ── 9. Add catch24 features ───────────────────────────────────────────────
    if add_catch24:
        df = add_catch24_features(
            df,
            window=WINDOW_CATCH22,
            step=STEP_CATCH22,
            time_interval="h",
            country=iso_objective,
        )
        # Drop first `window` days which have no catch24 lookback
        cutoff = df.index.min() + pd.Timedelta(days=WINDOW_CATCH22)
        df = df[df.index >= cutoff]

    # ── Final trim: drop last H rows with NaN from negative shift ────────────
    if add_future_time_features or add_future_meteo:
        df = df.iloc[:-H]
        print(
            f"  Dropped last {H} rows (future feature negative shift of {target_day_distance} days)"
        )

    return df
