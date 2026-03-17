import pandas as pd
pd.set_option('display.max_columns', None)
from power_forecast.logic.models.registry import save_scaler, load_scaler
from power_forecast.params import *

def preproc_arima(df, column):

    #cleaned from ancient nan values
    df_nonan = align_start_to_column(df=df, column=column)

    #cleaned from outliers
    df_clean = replace_outliers_with_interpolation(df_nonan, limit_low=-350, limit_high=2000)

    #extract one country
    df_country = df_clean[column]

    #resample for the mean of a day
    df_country_day = df_country.resample('D').mean()

    return df_country_day

def preproc_histxgb_train(df: pd.DataFrame, column: pd.Series, split_train_ratio: float, split_val_ratio: float):

    # # ── Step 1: build DataFrame──────────────────────────────────────────────
    # df = build_feature_dataframe('raw_data/all_countries.csv', load_from_pickle=False)

    # ── Step 2: X, y──────────────────────────────────────────────
    df['target'] = df[column]
    # df = df.dropna()
    y = df['target']
    X = df.drop(columns=['target'])

    # ── Step 3: scaler X ──────────────────────────────────────────────
    X, scaler = standard_scaling(X)
    save_scaler(scaler, scaler_name='HistXGB_scaler')

    # ── Step 4: definir  train, val, test ──────────────────────────────────────────────

    train_end = int(len(X) * split_train_ratio)
    val_end = int(len(X) * split_val_ratio)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def preproc_histxgb_X_new(df: pd.DataFrame, column: pd.Series):

    # # ── Step 1: build DataFrame──────────────────────────────────────────────
    # df = build_feature_dataframe('raw_data/all_countries.csv', load_from_pickle=False)

    # ── Step 2: X, y──────────────────────────────────────────────
    df['target'] = df[column]
    # df = df.dropna()
    X = df.drop(columns=['target'])

    # ── Step 3: scaler X ──────────────────────────────────────────────
    scaler = load_scaler(scaler_name='HistXGB_scaler')
    X = standard_scaling_X_new(X, scaler)

    return X

import pandas as pd
from sklearn.preprocessing import StandardScaler

def standard_scaling(df):
    scaler = StandardScaler().set_output(transform='pandas')
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler

def standard_scaling_X_new(df, scaler):
    df_scaled = scaler.transform(df)
    return df_scaled


