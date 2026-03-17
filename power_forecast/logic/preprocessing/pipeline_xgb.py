import numpy as np
import pandas as pd
from power_forecast.params import *
from power_forecast.logic.get_data.build_dataframe import (
    build_common_dataframe,
    add_features_XGB,
)
from power_forecast.logic.preprocessing.train_test_split import (
    train_test_split_general,
    train_test_split_XGB_optimized,
)
from power_forecast.logic.preprocessing.split_X_y_standardize import (
    X_y_standardizer_with_val_XGB,
    X_y_standardizer_XGB,
)


## PARAMÈTRES DE STRATÉGIE D'ENTRAÎNEMENT
## ==============================================================
max_train_test_split = True
## max_train_test_split (bool) :
##   - True  → Entraînement sur le maximum de données disponibles.
##              Le split train/test est ajusté automatiquement selon
##              l'objective_day. Aucun jeu de validation (X_val, y_val)
##              ne sera créé. Le modèle utilisera l'objective_day comme
##              X_new pour la prédiction finale, et les métriques seront
##              calculées en comparant y_true vs y_pred_xgb.
##
##   - False → Split train/test classique basé sur le cutoff_day
##              (01-01-2023 recommandé). Un jeu de validation
##              (X_val, y_val) sera également créé plus tard dans le code.
objective_day = pd.Timestamp("2024-03-20", tz="UTC")
## objective_day (Timestamp) :
##   Jour cible de prédiction. Utilisé comme X_new lorsque
##   max_train_test_split = True.

cutoff_day = pd.Timestamp("2023-10-01", tz="UTC")
## cutoff_day (Timestamp) :
##   Date de coupure pour le split train/test classique. Utilisé
##   uniquement lorsque max_train_test_split = False.
## ==============================================================

# Other inputs
prediction_horizon_days = 2
country_price_objective = "France"


df_common = build_common_dataframe(
    filepath="raw_data/all_countries.csv",
    country_objective=country_price_objective,
    target_day_distance=prediction_horizon_days,
    time_interval="h",
    keep_only_neighbors=True,
    add_meteo=True,
    add_crisis=True,
    add_entsoe=True,
)

df = add_features_XGB(
    df_common,
    country_objective=country_price_objective,
    target_day_distance=prediction_horizon_days,
    add_lag_frontiere=True,  # Si tu veux ajouter lags pour tout prix frontiere n_pays x (n_LAGS_XGB_FRONTIERE + 2 * ROLLING_WINDOWS_XGB_FRONTIERE)
    drop_initial_nans=True,
)

columns_xgb = df.columns
print(df.shape)
print(columns_xgb)


# if max_train_test_split = True il train jusqu'a derniere moment possible basè sur objective_day
if max_train_test_split:
    # XGB
    fold_train_xgb, fold_test_xgb = train_test_split_XGB_optimized(
        df=df,
        objective_day=objective_day,
        number_days_to_predict=prediction_horizon_days,
    )
if not max_train_test_split:
    # XGB
    fold_train_xgb, fold_test_xgb = train_test_split_general(df=df, cutoff=cutoff_day)

# if max_train_test_split = True pas besoin the X_val et y_val
if max_train_test_split:
    # Pas de validation, entraînement sur le maximum de données
    X_train, X_new, y_train, y_true = X_y_standardizer_XGB(
        fold_train=fold_train_xgb,
        fold_test=fold_test_xgb,
        country_objective=country_price_objective,
    )
    print("✅ Mode : entraînement maximal (max_train_test_split = True)")
    print(f"   X_train : {X_train.shape}")
    print(f"   y_train : {y_train.shape}")
    print(f"   X_new   : {X_new.shape}")
    print(f"   y_true  : {y_true.shape}")

if not max_train_test_split:
    # Avec jeu de validation, split chronologique sur cutoff_day
    X_train, X_val, X_test, y_train, y_val, y_test = X_y_standardizer_with_val_XGB(
        fold_train=fold_train_xgb,
        fold_test=fold_test_xgb,
        country_objective=country_price_objective,
        val_ratio=0.1,  # ajuste selon tes besoins
    )
    print("✅ Mode : split classique avec validation (max_train_test_split = False)")
    print(f"   X_train : {X_train.shape}")
    print(f"   y_train : {y_train.shape}")
    print(f"   X_val   : {X_val.shape}")
    print(f"   y_val   : {y_val.shape}")
    print(f"   X_test  : {X_test.shape}")
    print(f"   y_test  : {y_test.shape}")

if  max_train_test_split:
    print("🔍 Types et dtypes :")
    print(f"   X_train : {type(X_train)} | dtype: {X_train.dtype} | shape: {X_train.shape}")
    print(f"   y_train : {type(y_train)} | dtype: {y_train.dtype} | shape: {y_train.shape}")
    print(f"   X_new   : {type(X_new)}   | dtype: {X_new.dtype}   | shape: {X_new.shape}")
    print(f"   y_true  : {type(y_true)}  | dtype: {y_true.dtype}  | shape: {y_true.shape}")

if not max_train_test_split:
    print(f"   X_train : {type(X_train)} | dtype: {X_train.dtype} | shape: {X_train.shape}")
    print(f"   y_train : {type(y_train)} | dtype: {y_train.dtype} | shape: {y_train.shape}")
    print(f"   X_test  : {type(X_test)}  | dtype: {X_test.dtype}  | shape: {X_test.shape}")
    print(f"   y_true  : {type(y_true)}  | dtype: {y_true.dtype}  | shape: {y_true.shape}")
    print(f"   X_new   : {type(X_new)}   | dtype: {X_new.dtype}   | shape: {X_new.shape}")
    print(f"   X_val   : {type(X_val)}   | dtype: {X_val.dtype}   | shape: {X_val.shape}")
    print(f"   y_val   : {type(y_val)}   | dtype: {y_val.dtype}   | shape: {y_val.shape}")