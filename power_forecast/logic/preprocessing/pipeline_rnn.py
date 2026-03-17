import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from power_forecast.params import *
from power_forecast.logic.get_data.build_dataframe import (
    build_common_dataframe,
    add_features_RNN,
)
from power_forecast.logic.preprocessing.train_test_split import (
    train_test_split_general,
    train_test_split_RNN_optimized,
)
from power_forecast.logic.preprocessing.split_X_y_standardize import (
    get_X_y_vectorized_RNN,
    get_Xi_yi_single_sequence,
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
input_length = 14 * 24  # 3 weeks context fed to RNN
stride_sequences = 24 * 3  # doit etre plus haute que output length
prediction_horizon_days = 2
country_price_objective = "France"
prediction_length = prediction_horizon_days * 24  # predict 48h of target day

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

df = add_features_RNN(
    df=df_common,
    country_objective=country_price_objective,
    target_day_distance=prediction_horizon_days,
    add_future_time_features=True,
    add_future_meteo=True,
)

columns_rnn = df.columns
print(df.shape)


# if max_train_test_split = True il train jusqu'a derniere moment possible basè sur objective_day
if max_train_test_split:
    # RNN
    fold_train_rnn, fold_test_rnn = train_test_split_RNN_optimized(
        df=df,
        objective_day=objective_day,
        number_days_to_predict=prediction_horizon_days,
        input_length=input_length,  # 168h lookback
    )
if not max_train_test_split:
    # RNN
    fold_train_rnn, fold_test_rnn = train_test_split_general(df=df, cutoff=cutoff_day)


scaler = StandardScaler()

# ── Train ──────────────────────────────────────────────────────────────────
X_train, y_train = get_X_y_vectorized_RNN(
    fold=fold_train_rnn,
    feature_cols=fold_train_rnn.columns,
    country_objective=country_price_objective,
    stride=stride_sequences,
    input_length=input_length,
    output_length=prediction_length,
    scaler=scaler,
    fit_scaler=True,
)

# ── Test ───────────────────────────────────────────────────────────────────
X_new, y_true = get_X_y_vectorized_RNN(
    fold=fold_test_rnn,
    feature_cols=fold_test_rnn.columns,
    country_objective=country_price_objective,
    stride=stride_sequences,
    input_length=input_length,
    output_length=prediction_length,
    scaler=scaler,
    fit_scaler=False,
)

# # ── X_new : dernière séquence du fold_test pour prédiction ────────────────
# X_new = X_test[-1:]  # (1, input_length, n_features) -> deja bon dimension

if max_train_test_split:
    print("📐 Shapes finales :")
    print(f"    X_train: {X_train.shape} → (n_seq, input_length, n_features)")
    print(f"    y_train: {y_train.shape} → (n_seq, output_length)")
    print(f"    X_new: {X_new.shape} → (1, input_length, n_features)")
    print(f"    y_true: {y_true.shape}→ (n_seq, output_length)")

# ── Validation : split chronologique SUR LES SÉQUENCES (pas sur le fold brut)
if not max_train_test_split:
    val_ratio = 0.2
    split_idx = int(len(X_train) * (1 - val_ratio))

    X_val = X_train[split_idx:]  # séquences val → suivent chronologiquement train
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]  # on réduit X_train en conséquence
    y_train = y_train[:split_idx]
    print("📐 Shapes finales :")
    print(f"    X_train: {X_train.shape} → (n_seq, input_length, n_features)")
    print(f"    y_train: {y_train.shape} → (n_seq, output_length)")
    print(f"    X_val: {X_val.shape} → (n_seq, input_length, n_features)")
    print(f"    y_val: {y_val.shape} → (n_seq, output_length)")
    print(f"    X_test: {X_new.shape} → (1, input_length, n_features)")
    print(f"    y_test: {y_true.shape}→ (n_seq, output_length)")

input_shape=X_train.shape[1:]
output_length=y_train.shape[1]

if  max_train_test_split:
    print("🔍 Types et dtypes :")
    print(f"   X_train : {type(X_train)} | dtype: {X_train.dtype} | shape: {X_train.shape}")
    print(f"   y_train : {type(y_train)} | dtype: {y_train.dtype} | shape: {y_train.shape}")
    print(f"   X_new   : {type(X_new)}   | dtype: {X_new.dtype}   | shape: {X_new.shape}")
    print(f"   y_true  : {type(y_true)}  | dtype: {y_true.dtype}  | shape: {y_true.shape}")

if not max_train_test_split:
    print(f"   X_train : {type(X_train)} | dtype: {X_train.dtype} | shape: {X_train.shape}")
    print(f"   y_train : {type(y_train)} | dtype: {y_train.dtype} | shape: {y_train.shape}")
    print(f"   X_test   : {type(X_new)}   | dtype: {X_new.dtype}   | shape: {X_new.shape}")
    print(f"   y_test  : {type(y_true)}  | dtype: {y_true.dtype}  | shape: {y_true.shape}")
    print(f"   X_val   : {type(X_val)}   | dtype: {X_val.dtype}   | shape: {X_val.shape}")
    print(f"   y_val   : {type(y_val)}   | dtype: {y_val.dtype}   | shape: {y_val.shape}")