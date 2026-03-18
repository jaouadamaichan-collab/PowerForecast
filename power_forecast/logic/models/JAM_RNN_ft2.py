import numpy as np
import pandas as pd
from power_forecast.logic.utils.graphs import plot_history, plot_history_loss_is_mae
from typing import Dict, List, Tuple, Sequence
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from tensorflow.keras import models, layers, Input, optimizers, metrics
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import Normalization
from keras.callbacks import EarlyStopping
from keras.layers import Lambda
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
max_train_test_split = False
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
#output_length
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



if max_train_test_split:
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
    print("📐 Shapes finales :")
    print(f"    X_train: {X_train.shape} → (n_seq, input_length, n_features)")
    print(f"    y_train: {y_train.shape} → (n_seq, output_length)")
    print(f"    X_new: {X_new.shape} → (1, input_length, n_features)")
    print(f"    y_true: {y_true.shape}→ (n_seq, output_length)")

# ── Validation : split chronologique SUR LES SÉQUENCES (pas sur le fold brut)
if not max_train_test_split:

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
    X_test, y_test = get_X_y_vectorized_RNN(
        fold=fold_test_rnn,
        feature_cols=fold_test_rnn.columns,
        country_objective=country_price_objective,
        stride=stride_sequences,
        input_length=input_length,
        output_length=prediction_length,
        scaler=scaler,
        fit_scaler=False,
    )
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

# # ── X_new : dernière séquence du fold_test pour prédiction ────────────────
X_new = X_test[-1:]  # (1, input_length, n_features) -> deja bon dimension


def init_model(X_train: np.ndarray, y_train: np.ndarray) -> tf.keras.Model:
    """
    Construit et compile le modèle LSTM pour la prévision de série temporelle.

    Architecture :
        1. Couche de normalisation — standardise chaque feature selon les statistiques d'entraînement.
        2. Couche LSTM (64 unités, tanh) — capture les dépendances temporelles.
           Régularisation L1L2 réduite (0.01/0.01) pour limiter la sur-contrainte. [FIX-2]
        3. Couche Dense de sortie — projection linéaire sur output_length prédictions.

    Compilation :
        - Perte : MSE
        - Optimiseur : Adam avec gradient clipping (clipnorm=1.0) [FIX-1]
        - Métrique : MAE

    Args:
        X_train (np.ndarray): Entrées d'entraînement (n_samples, input_length, n_features).
        y_train (np.ndarray): Cibles d'entraînement normalisées (n_samples, output_length, 1).

    Returns:
        tf.keras.Model: Modèle Keras compilé, prêt pour l'entraînement.
    """
    normalizer = Normalization()
    normalizer.adapt(X_train)

    model = models.Sequential()
    model.add(Input(shape=X_train[0].shape))
    model.add(normalizer)
    model.add(layers.LSTM(
        64,
        activation='tanh',
        return_sequences=False,
        # [FIX-2] L1L2 réduit : 0.05 → 0.01 pour limiter la sur-contrainte
        kernel_regularizer=L1L2(l1=0.01, l2=0.01)
    ))
    output_length = y_train.shape[1]
    model.add(layers.Dense(output_length, activation='linear'))

    # [FIX-1] clipnorm=1.0 : coupe les gradients dont la norme dépasse 1.0
    # Élimine les explosions de gradient (val_loss 31 → 417 observées epoch 13→14)
    adam = optimizers.Adam(learning_rate=0.005, clipnorm=1.0)
    model.compile(loss='mae', optimizer=adam, metrics=["mse"])

    return model

def fit_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train_scaled: np.ndarray,
    verbose: int = 1
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Entraîne le modèle LSTM avec arrêt anticipé (early stopping) sur la perte de validation.

    Configuration d'entraînement :
        - 10% des données d'entraînement réservées pour la validation.
        - Pas de mélange (shuffle=False) pour préserver l'ordre temporel.
        - Early stopping sur val_loss [FIX-3], patience=10, restaure les meilleurs poids.

    Args:
        model (tf.keras.Model): Modèle Keras compilé (issu de `init_model`).
        X_train (np.ndarray): Séquences d'entrée d'entraînement.
        y_train_scaled (np.ndarray): Cibles d'entraînement normalisées. [FIX-5]
        verbose (int): Niveau de verbosité (0=silencieux, 1=barre de progression).

    Returns:
        Tuple[tf.keras.Model, History]: (modèle entraîné, historique d'entraînement)
    """
    es = EarlyStopping(
        # [FIX-3] val_loss est plus stable que val_mae pour l'early stopping
        monitor="val_loss",
        patience=10,
        mode="min",
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val,y_val ),
        shuffle=False,
        batch_size=16,
        epochs=100,
        callbacks=[es],
        verbose=verbose
    )

    return model, history


# ====================================
# 1 - Initialisation
# ====================================
model = init_model(X_train, y_train)
model.summary()

# ====================================
# 2 - Entraînement
# ====================================
model, history = fit_model(model, X_train, y_train)
plot_history_loss_is_mae(history)

# ====================================
# 3 - Évaluation sur le test set
#     Les prédictions sont dénormalisées avant calcul de la MAE [FIX-5]
# ====================================
# Evaluate model on test set
loss, mse = model.evaluate(X_test, y_test, verbose=1)


y_pred = model.predict(X_test)
mae_lstm = np.mean(np.abs(y_pred - y_test))
print(f"The LSTM MAE on the test set is equal to {round(mae_lstm, 2)}")


def init_baseline() -> tf.keras.Model:
    """
    Construit et compile un modèle baseline naïf de type "dernière valeur observée".

    Ce modèle sert de borne inférieure de performance : si le LSTM ne fait pas mieux,
    il n'a rien appris d'utile.

    Returns:
        tf.keras.Model: Modèle Keras compilé retournant la dernière valeur observée de TARGET.
    """
    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x[:, -1, 1, None]))

    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model


baseline_model   = init_baseline()
baseline_score   = baseline_model.evaluate(X_test, y_test)

print(f"- The Baseline MAE on the test set is equal to {round(baseline_score[1], 2)}")
print(f"- The LSTM MAE on the test set is equal to {round(mae_lstm, 2)}")
print(f"Improvement of the LSTM over the baseline : {round((1 - (mae_lstm / baseline_score[1])) * 100, 2)} %")

print(f'N_FEATURES = {len(df.columns)}')
print(f'SEQUENCE_LENGTH = {input_length}')
print(f'SEQUENCE_STRIDE = {stride_sequences}')
print(f'N_TRAIN = {X_train.shape[0]}')
print(f'N_TEST = {X_test.shape[0]}')
print(f'INPUT_LENGTH = {input_length}')
print(f'OUTPUT_LENGTH = {prediction_length}')


def get_folds(
    df: pd.DataFrame,
    fold_length: int,
    fold_stride: int
) -> List[pd.DataFrame]:
    """
    Parcourt un DataFrame de série temporelle pour en extraire des folds de longueur fixe.

    Args:
        df           : DataFrame complet de la série temporelle (n_pas, n_features).
        fold_length  : Nombre de lignes dans chaque fold.
        fold_stride  : Nombre de lignes à avancer entre deux folds consécutifs.

    Returns:
        List[pd.DataFrame] : Liste de DataFrames, chacun représentant un fold.
    """
    folds = []
    for idx in range(0, len(df), fold_stride):
        if (idx + fold_length) > len(df):
            break
        folds.append(df.iloc[idx:idx + fold_length])
    return folds


def cross_validate_RNN(
    df: pd.DataFrame,
    fold_length: int,
    fold_stride: int,
    train_test_ratio: float,
    feature_cols: list,
    country_objective: str,
    stride_sequences: int,
    input_length: int = INPUT_LENGTH,
    output_length: int = OUTPUT_LENGTH,
) -> Tuple[List[float], List[float]]:
    """
    Validation croisée temporelle pour un modèle RNN.

    Pour chaque fold :
        1. Split train / test chronologique via train_test_ratio.
        2. Construction des séquences X/y par fenêtre glissante (leakage-free).
        3. Standardisation de X (fit sur train, transform sur test).
        4. Si max_train_test_split=False → split val chronologique sur les séquences.
        5. Entraînement du modèle RNN avec early stopping.
        6. Évaluation MAE sur y_true vs y_pred.

    Args:
        df                : DataFrame complet de la série temporelle.
        fold_length       : Longueur de chaque fold.
        fold_stride       : Pas entre deux folds.
        train_test_ratio  : Proportion train (ex: 0.8).
        feature_cols      : Colonnes features pour X.
        country_objective : Pays cible (clé dans VILLE_TO_ISO).
        stride_sequences  : Stride entre séquences (>= output_length).
        input_length      : Longueur des séquences X.
        output_length     : Longueur des séquences y.

    Returns:
        Tuple[List[float], List[float]]:
            - list_mae_baseline : MAE baseline naïf par fold.
            - list_mae_rnn      : MAE RNN par fold.
    """
    list_mae_baseline = []
    list_mae_rnn      = []

    folds = get_folds(df, fold_length, fold_stride)

    for fold_id, fold in enumerate(folds):
        print(f"\n{'='*55}")
        print(f"  FOLD {fold_id + 1} / {len(folds)}")
        print(f"{'='*55}")

        # ── 1. Split train / test chronologique ───────────────────────────
        split_idx  = int(len(fold) * train_test_ratio)
        fold_train = fold.iloc[:split_idx]
        fold_test  = fold.iloc[split_idx:]

        # ── 2. Construction des séquences + standardisation X ─────────────
        scaler = StandardScaler()

        X_train, y_train = get_X_y_vectorized_RNN(
            fold              = fold_train,
            feature_cols      = feature_cols,
            country_objective = country_objective,
            stride            = stride_sequences,
            input_length      = input_length,
            output_length     = output_length,
            scaler            = scaler,
            fit_scaler        = True,
        )

        X_test, y_true = get_X_y_vectorized_RNN(
            fold              = fold_test,
            feature_cols      = feature_cols,
            country_objective = country_objective,
            stride            = stride_sequences,
            input_length      = input_length,
            output_length     = output_length,
            scaler            = scaler,
            fit_scaler        = False,
        )

        # ── 3. Split val chronologique sur les séquences (si demandé) ─────
        if not max_train_test_split:
            val_ratio = 0.2
            split_val = int(len(X_train) * (1 - val_ratio))
            X_val    = X_train[split_val:]
            y_val    = y_train[split_val:]
            X_train  = X_train[:split_val]
            y_train  = y_train[:split_val]
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        print(f"    X_train : {X_train.shape} | y_train : {y_train.shape}")
        if not max_train_test_split:
            print(f"    X_val   : {X_val.shape}   | y_val   : {y_val.shape}")
        print(f"    X_test  : {X_test.shape}  | y_true  : {y_true.shape}")

        # ── 4. Baseline naïf ──────────────────────────────────────────────
        # dernière valeur connue de X répétée sur output_length
        y_pred_baseline = np.repeat(
            X_test[:, -1, feature_cols.get_loc(VILLE_TO_ISO[country_objective])].reshape(-1, 1),
            output_length, axis=1
        )
        mae_baseline = float(np.mean(np.abs(y_pred_baseline - y_true)))
        list_mae_baseline.append(mae_baseline)
        print(f"\n  MAE baseline fold {fold_id + 1} : {mae_baseline:.2f}")

        # ── 5. Entraînement RNN ───────────────────────────────────────────
        model = init_model(X_train, y_train)

        es = EarlyStopping(
            monitor="val_loss" if validation_data else "loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
        )

        model.fit(
            X_train, y_train,
            validation_data = validation_data,
            shuffle          = False,         # timeseries → pas de shuffle
            batch_size       = 16,
            epochs           = 100,
            callbacks        = [es],
            verbose          = 0,
        )

        # ── 6. Évaluation ─────────────────────────────────────────────────
        y_pred   = model.predict(X_test, verbose=0)
        mae_rnn  = float(np.mean(np.abs(y_pred - y_true)))
        list_mae_rnn.append(mae_rnn)

        improvement = (1 - mae_rnn / mae_baseline) * 100
        print(f"  MAE RNN   fold {fold_id + 1} : {mae_rnn:.2f}")
        print(f"  Amélioration vs baseline     : {improvement:.2f} %")

    print(f"\n{'='*55}")
    print(f"  MAE baseline moyenne : {np.mean(list_mae_baseline):.2f}")
    print(f"  MAE RNN moyenne      : {np.mean(list_mae_rnn):.2f}")
    print(f"{'='*55}\n")

    return list_mae_baseline, list_mae_rnn

# def get_folds(
#     df: pd.DataFrame,
#     fold_length: int,
#     fold_stride: int) -> List[pd.DataFrame]:
#     """
#     Parcourt un DataFrame de série temporelle pour en extraire des folds de longueur fixe.

#     Chaque fold est une fenêtre contiguë de `fold_length` lignes extraite du DataFrame.
#     La fenêtre avance de `fold_stride` lignes à chaque itération.
#     Toute fenêtre qui dépasserait la fin du DataFrame est ignorée.

#     Args:
#         df (pd.DataFrame): Le DataFrame complet de la série temporelle, de forme (n_pas, n_features).
#         fold_length (int): Nombre de lignes (pas de temps) dans chaque fold.
#         fold_stride (int): Nombre de lignes à avancer entre deux folds consécutifs.

#     Returns:
#         List[pd.DataFrame]: Liste de DataFrames, chacun représentant un fold.
#     """
#     folds = []
#     for idx in range(0, len(df), fold_stride):
#         if (idx + fold_length) > len(df):
#             break
#         fold = df.iloc[idx:idx + fold_length, :]
#         folds.append(fold)
#     return folds

# def cross_validate_baseline_and_lstm():
#     """
#     Effectue une validation croisée temporelle en comparant le baseline et le modèle LSTM.

#     Pour chaque fold :
#         1. Division en train/test.
#         2. Normalisation de y_train ; transformation (sans re-fit) de y_test. [FIX-5]
#         3. Évaluation du baseline naïf.
#         4. Entraînement LSTM avec early stopping robuste [FIX-3] + gradient clipping [FIX-1].
#         5. Dénormalisation des prédictions avant calcul de la MAE.

#     Returns:
#         Tuple[List[float], List[float]]:
#             - list_of_mae_baseline_model  : MAE du baseline pour chaque fold.
#             - list_of_mae_recurrent_model : MAE du LSTM pour chaque fold.
#     """
#     list_of_mae_baseline_model  = []
#     list_of_mae_recurrent_model = []

#     folds = get_folds(df_selected, FOLD_LENGTH, FOLD_STRIDE)

#     for fold_id, fold in enumerate(folds):

#         # 1 - Division train / test
#         (fold_train, fold_test) = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)

#         X_train, y_train = get_X_y(fold_train, N_TRAIN, INPUT_LENGTH, OUTPUT_LENGTH)
#         X_test,  y_test  = get_X_y(fold_test,  N_TEST,  INPUT_LENGTH, OUTPUT_LENGTH)

#         # [FIX-5] Normalisation de y — fitté sur y_train uniquement
#         target_scaler   = TargetScaler()
#         y_train_scaled  = target_scaler.fit_transform(y_train)
#         y_test_scaled   = target_scaler.transform(y_test)
#         y_test_original = target_scaler.inverse_transform(y_test_scaled)

#         # 2a - Baseline évalué sur les valeurs originales (non normalisées)
#         baseline_model = init_baseline()
#         mae_baseline   = baseline_model.evaluate(X_test, y_test_original, verbose=0)[1]
#         list_of_mae_baseline_model.append(mae_baseline)
#         print("-" * 50)
#         print(f"MAE baseline fold n°{fold_id} = {round(mae_baseline, 2)}")

#         # 2b - LSTM : entraînement sur y normalisé
#         model = init_model(X_train, y_train_scaled)

#         es = EarlyStopping(
#             # [FIX-3] monitor=val_loss + patience=5 (vs patience=2 + val_mae avant)
#             # val_loss est plus stable ; patience=2 stoppait trop tôt sur les folds volatils
#             monitor="val_loss",
#             mode="min",
#             patience=5,
#             restore_best_weights=True
#         )

#         model.fit(
#             X_train, y_train_scaled,
#             validation_split=0.1,
#             shuffle=False,
#             batch_size=16,
#             epochs=100,
#             callbacks=[es],
#             verbose=0
#         )

#         # [FIX-5] Dénormalisation avant calcul de la MAE
#         y_pred_scaled = model.predict(X_test, verbose=0)
#         y_pred        = target_scaler.inverse_transform(y_pred_scaled)
#         mae_lstm      = float(np.mean(np.abs(y_pred - y_test_original)))
#         list_of_mae_recurrent_model.append(mae_lstm)

#         print(f"MAE LSTM fold n°{fold_id} = {round(mae_lstm, 2)}")
#         print(f"Improvement over baseline: {round((1 - (mae_lstm / mae_baseline)) * 100, 2)} %\n")

#     return list_of_mae_baseline_model, list_of_mae_recurrent_model


# mae_baselines, mae_lstms = cross_validate_baseline_and_lstm()
