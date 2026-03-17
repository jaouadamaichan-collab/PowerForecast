"""
Power Forecast - Modèle LSTM pour la Prévision de Consommation Électrique
==========================================================================
Ce notebook entraîne et évalue un réseau de neurones récurrent LSTM (Long Short-Term Memory)
pour prédire la consommation électrique de la France (FRA), à partir d'un jeu de données
multi-pays.

Le pipeline comprend :
    - Découpage de la série temporelle en folds
    - Séparation train/test de chaque fold
    - Génération de séquences (aléatoire et par pas fixe)
    - Entraînement du modèle LSTM avec arrêt anticipé (early stopping)
    - Comparaison avec un modèle baseline "dernière valeur observée"
    - Validation croisée sur l'ensemble des folds

CORRECTIFS APPLIQUÉS :
    [FIX-1] Gradient clipping (clipnorm=1.0) sur Adam → élimine les explosions de gradient
    [FIX-2] Régularisation L1L2 réduite (0.05→0.01) → moins de sur-contrainte
    [FIX-3] patience=5 + monitor='val_loss' en cross-validation → meilleure convergence
    [FIX-4] Feature rolling z-score sur la target → détection du régime de crise
    [FIX-5] Normalisation séparée de y_train/y_test → stabilise la loss sur folds extrêmes
"""

import pandas as pd
from power_forecast.logic.get_data.build_dataframe import build_common_dataframe, add_features_RNN
# %load_ext autoreload
# %autoreload 2
pd.set_option('display.max_columns', None)

# Chargement du DataFrame de features à partir d'un CSV multi-pays.
# `load_from_pickle=False` force un rechargement complet plutôt que d'utiliser un cache.

df_common = build_common_dataframe(
    filepath="raw_data/all_countries.csv",
    country_objective="France",
    target_day_distance=2,
    time_interval="h",
    keep_only_neighbors=True,
    add_meteo=True,
    add_crisis=True,
    add_entsoe=True,
)

df_rnn = add_features_RNN(
    df=df_common,
    country_objective="France",
    target_day_distance=2,
    add_future_time_features=True,
    add_future_meteo=True,
)

columns_rnn = df_rnn.columns
print(df_rnn.shape)
print(columns_rnn)

df= df_rnn

from typing import Dict, List, Tuple, Sequence
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from tensorflow.keras import models, layers, Input, optimizers, metrics
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import Normalization
from keras.callbacks import EarlyStopping
from keras.layers import Lambda


# ================================================================= #
# [FIX-4] FEATURE : ROLLING Z-SCORE SUR LA TARGET                  #
# ================================================================= #
# Ajoute un z-score glissant (fenêtre 7 jours = 168h) sur la colonne
# TARGET. Cette feature indique au modèle si le prix actuel est dans un
# régime "normal" ou "extrême" (crise énergétique, pic de demande...).
# Un |z| > 2 signale un régime hors-distribution que le LSTM doit traiter
# différemment — sans cette information, il extrapole aveuglément.

def add_rolling_zscore(df: pd.DataFrame, target: str, window: int = 168) -> pd.DataFrame:
    """
    Ajoute une feature de z-score glissant sur la colonne cible.

    Le z-score est calculé sur une fenêtre mobile de `window` pas de temps.
    Il mesure l'écart de la valeur courante par rapport à la distribution
    locale récente, permettant au modèle de détecter les régimes extrêmes.

    Args:
        df (pd.DataFrame): Le DataFrame complet de la série temporelle.
        target (str): Nom de la colonne cible (ex. 'FRA').
        window (int): Taille de la fenêtre glissante en pas de temps (défaut : 168h = 7 jours).

    Returns:
        pd.DataFrame: Le DataFrame avec une colonne supplémentaire `{target}_zscore`.
    """
    rolling_mean = df[target].rolling(window=window, min_periods=1).mean()
    rolling_std  = df[target].rolling(window=window, min_periods=1).std().replace(0, 1)
    df[f'{target}_zscore'] = (df[target] - rolling_mean) / rolling_std
    df[f'{target}_zscore'] = df[f'{target}_zscore'].fillna(0)
    return df


df = add_rolling_zscore(df, target='FRA', window=168)


# ================================================================= #
# 2. FEATURE SELECTION PAR LASSO                                    #
# ================================================================= #

def lasso_feature_selection(df: pd.DataFrame, target: str,
                             alpha: float = 0.01) -> List[str]:
    features = [c for c in df.columns if c != target]
    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_scaled, y)

    selected = [features[i] for i, coef in enumerate(lasso.coef_) if coef != 0]
    print(f"\nLasso feature selection (alpha={alpha}) :")
    print(f"  {len(features)} features initiales → {len(selected)} features sélectionnées")
    print(f"  Features conservées : {selected}")
    return selected


selected_features = lasso_feature_selection(df, target='FRA', alpha=0.01)

# Note : FRA_zscore est conservée si Lasso la juge pertinente.
# Si elle est éliminée, on la force manuellement car elle est structurellement utile.
if 'FRA_zscore' not in selected_features:
    selected_features.append('FRA_zscore')
    print("[FIX-4] 'FRA_zscore' forcée dans les features (non sélectionnée par Lasso).")

df_selected = df[selected_features + ['FRA']]
print(f"\nShape après feature selection : {df_selected.shape}")


# --------------------------------------------------- #
# Configuration globale du jeu de données             #
# --------------------------------------------------- #

TARGET          = 'FRA'
N_FEATURES      = df_selected.shape[1]

FOLD_LENGTH      = 24 * 365 * 4
FOLD_STRIDE      = 24 * 182
TRAIN_TEST_RATIO = 0.9

INPUT_LENGTH    = 24 * 7
OUTPUT_LENGTH   = 1
SEQUENCE_STRIDE = 24
DAY_AHEAD_GAP   = 0

print(f"N_FEATURES = {N_FEATURES} | INPUT_LENGTH = {INPUT_LENGTH}h = {INPUT_LENGTH//24} jours")
print(f"FOLD_LENGTH = {FOLD_LENGTH}h = {FOLD_LENGTH//24/365:.1f} ans")


def get_folds(
    df: pd.DataFrame,
    fold_length: int,
    fold_stride: int) -> List[pd.DataFrame]:
    """
    Parcourt un DataFrame de série temporelle pour en extraire des folds de longueur fixe.

    Chaque fold est une fenêtre contiguë de `fold_length` lignes extraite du DataFrame.
    La fenêtre avance de `fold_stride` lignes à chaque itération.
    Toute fenêtre qui dépasserait la fin du DataFrame est ignorée.

    Args:
        df (pd.DataFrame): Le DataFrame complet de la série temporelle, de forme (n_pas, n_features).
        fold_length (int): Nombre de lignes (pas de temps) dans chaque fold.
        fold_stride (int): Nombre de lignes à avancer entre deux folds consécutifs.

    Returns:
        List[pd.DataFrame]: Liste de DataFrames, chacun représentant un fold.
    """
    folds = []
    for idx in range(0, len(df), fold_stride):
        if (idx + fold_length) > len(df):
            break
        fold = df.iloc[idx:idx + fold_length, :]
        folds.append(fold)
    return folds


folds = get_folds(df_selected, FOLD_LENGTH, FOLD_STRIDE)

print(f'The function generated {len(folds)} folds.')
print(f'Each fold has a shape equal to {folds[0].shape}.')

fold = folds[0]


def train_test_split(fold: pd.DataFrame,
                     train_test_ratio: float,
                     input_length: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divise un fold en un ensemble d'entraînement et un ensemble de test.

    L'ensemble d'entraînement contient les premières `train_test_ratio` lignes du fold.
    L'ensemble de test commence `input_length` lignes avant la fin de l'entraînement,
    afin que le modèle dispose d'une fenêtre d'entrée complète dès sa première prédiction.
    Ce léger chevauchement est intentionnel et ne constitue pas une fuite de données.

    Args:
        fold (pd.DataFrame): Un fold de pas de temps, de forme (fold_length, n_features).
        train_test_ratio (float): Proportion de lignes allouées à l'entraînement (ex : 0.7).
        input_length (int): Nombre de pas de temps nécessaires pour former une séquence X_i.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (fold_train, fold_test)
    """
    last_train_idx = round(train_test_ratio * len(fold))
    fold_train = fold.iloc[0:last_train_idx, :]

    first_test_idx = last_train_idx - input_length
    fold_test = fold.iloc[first_test_idx:, :]

    return (fold_train, fold_test)


(fold_train, fold_test) = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)

print(f'N_FEATURES = {N_FEATURES}')
print(f'INPUT_LENGTH = {INPUT_LENGTH} timesteps = {int(INPUT_LENGTH)/24} days')



# ================================================================= #
# [FIX-5] NORMALISATION SÉPARÉE DE Y                               #
# ================================================================= #
# Sur les folds de crise (folds 20+), la target FRA explose à des
# valeurs 5-10x supérieures aux périodes normales. Cela déforme la MSE
# et rend l'entraînement instable. On normalise y séparément avec un
# StandardScaler fitté uniquement sur y_train, puis on dénormalise
# les prédictions pour retrouver les unités originales.

class TargetScaler:
    """
    Normalise et dénormalise la target (y) indépendamment des features (X).

    Nécessaire quand la distribution de la target varie fortement entre
    les folds (ex. crise énergétique). Fitter sur y_train uniquement
    pour éviter toute fuite de données depuis y_test.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fitte le scaler sur y et retourne y normalisé."""
        shape = y.shape
        y_flat = y.reshape(-1, 1)
        y_scaled = self.scaler.fit_transform(y_flat)
        self._is_fitted = True
        return y_scaled.reshape(shape)

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Normalise y sans re-fitter (utilise les stats de fit_transform)."""
        assert self._is_fitted, "Appeler fit_transform avant transform."
        shape = y.shape
        return self.scaler.transform(y.reshape(-1, 1)).reshape(shape)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Dénormalise y pour retrouver les unités d'origine."""
        assert self._is_fitted, "Appeler fit_transform avant inverse_transform."
        shape = y_scaled.shape
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(shape)


def get_Xi_yi(
    fold: pd.DataFrame,
    input_length: int,
    output_length: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrait une paire (entrée, cible) depuis un fold en choisissant un point de départ aléatoire.

    La cible `y_i` est positionnée 24 heures après la fin de `X_i`, simulant un
    scénario de prévision à horizon J+1 (day-ahead forecasting).

    Args:
        fold (pd.DataFrame): Un seul fold de la série temporelle.
        input_length (int): Nombre de pas de temps dans la séquence d'entrée X_i.
        output_length (int): Nombre de pas de temps à prédire (y_i).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (X_i, y_i)
    """
    first_possible_start = 0
    last_possible_start = len(fold) - (input_length + output_length + 24) + 1
    random_start = np.random.randint(first_possible_start, last_possible_start)

    X_i = fold.iloc[random_start:random_start + input_length]
    y_i = fold.iloc[random_start + input_length + 24:
                    random_start + input_length + output_length + 24][[TARGET]]

    return (X_i, y_i)


def get_X_y(
    fold: pd.DataFrame,
    number_of_sequences: int,
    input_length: int,
    output_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit un jeu de données en échantillonnant aléatoirement des paires (X_i, y_i) dans un fold.

    Args:
        fold (pd.DataFrame): Le fold depuis lequel échantillonner.
        number_of_sequences (int): Nombre total de paires (X, y) à générer.
        input_length (int): Longueur de chaque séquence d'entrée.
        output_length (int): Longueur de chaque séquence cible.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) de formes
            (number_of_sequences, input_length, n_features) et
            (number_of_sequences, output_length, 1)
    """
    X, y = [], []
    for i in range(number_of_sequences):
        (Xi, yi) = get_Xi_yi(fold, input_length, output_length)
        X.append(Xi)
        y.append(yi)
    return np.array(X), np.array(y)


N_TRAIN = 504
N_TEST  = 48



def get_X_y_strides(fold: pd.DataFrame, input_length: int, output_length: int,
                    sequence_stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit un jeu de données en faisant glisser une fenêtre à pas fixe sur le fold.

    Args:
        fold (pd.DataFrame): Un seul fold de la série temporelle.
        input_length (int): Nombre de pas de temps dans chaque fenêtre X_i.
        output_length (int): Nombre de pas de temps dans chaque cible y_i.
        sequence_stride (int): Nombre de pas de temps entre le début de deux séquences consécutives.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y)
    """
    X, y = [], []
    for i in range(0, len(fold), sequence_stride):
        if (i + input_length + output_length) >= len(fold):
            break
        X_i = fold.iloc[i:i + input_length, :]
        y_i = fold.iloc[i + input_length:i + input_length + output_length, :][[TARGET]]
        X.append(X_i)
        y.append(y_i)
    return (np.array(X), np.array(y))


X_train, y_train = get_X_y_strides(fold_train, INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)
X_test,  y_test  = get_X_y_strides(fold_test,  INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

# [FIX-5] Normalisation de y — fitté sur train uniquement
target_scaler = TargetScaler()
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled  = target_scaler.transform(y_test)

print(X_train.shape)
print(y_train_scaled.shape)


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
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model


def plot_history(history: tf.keras.callbacks.History):
    """
    Affiche les courbes de perte (MSE) et de métrique (MAE) pour l'entraînement et la validation.

    Args:
        history (tf.keras.callbacks.History): L'objet History retourné par `model.fit()`.

    Returns:
        np.ndarray: Tableau de deux objets Axes matplotlib.
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))

    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('MSE')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='best')
    ax[0].grid(axis="x", linewidth=0.5)
    ax[0].grid(axis="y", linewidth=0.5)

    ax[1].plot(history.history['mae'])
    ax[1].plot(history.history['val_mae'])
    ax[1].set_title('MAE')
    ax[1].set_ylabel('MAE')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='best')
    ax[1].grid(axis="x", linewidth=0.5)
    ax[1].grid(axis="y", linewidth=0.5)

    return ax


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
        validation_split=0.1,
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
model = init_model(X_train, y_train_scaled)
model.summary()

# ====================================
# 2 - Entraînement
# ====================================
model, history = fit_model(model, X_train, y_train_scaled)
plot_history(history)

# ====================================
# 3 - Évaluation sur le test set
#     Les prédictions sont dénormalisées avant calcul de la MAE [FIX-5]
# ====================================
y_pred_scaled = model.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test_original = target_scaler.inverse_transform(y_test_scaled)

mae_lstm = np.mean(np.abs(y_pred - y_test_original))
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
baseline_score   = baseline_model.evaluate(X_test, y_test_original)

print(f"- The Baseline MAE on the test set is equal to {round(baseline_score[1], 2)}")
print(f"- The LSTM MAE on the test set is equal to {round(mae_lstm, 2)}")
print(f"Improvement of the LSTM over the baseline : {round((1 - (mae_lstm / baseline_score[1])) * 100, 2)} %")

print(f'N_FEATURES = {N_FEATURES}')
print(f'FOLD_LENGTH = {FOLD_LENGTH}')
print(f'FOLD_STRIDE = {FOLD_STRIDE}')
print(f'TRAIN_TEST_RATIO = {TRAIN_TEST_RATIO}')
print(f'N_TRAIN = {N_TRAIN}')
print(f'N_TEST = {N_TEST}')
print(f'INPUT_LENGTH = {INPUT_LENGTH}')
print(f'OUTPUT_LENGTH = {OUTPUT_LENGTH}')


def cross_validate_baseline_and_lstm():
    """
    Effectue une validation croisée temporelle en comparant le baseline et le modèle LSTM.

    Pour chaque fold :
        1. Division en train/test.
        2. Normalisation de y_train ; transformation (sans re-fit) de y_test. [FIX-5]
        3. Évaluation du baseline naïf.
        4. Entraînement LSTM avec early stopping robuste [FIX-3] + gradient clipping [FIX-1].
        5. Dénormalisation des prédictions avant calcul de la MAE.

    Returns:
        Tuple[List[float], List[float]]:
            - list_of_mae_baseline_model  : MAE du baseline pour chaque fold.
            - list_of_mae_recurrent_model : MAE du LSTM pour chaque fold.
    """
    list_of_mae_baseline_model  = []
    list_of_mae_recurrent_model = []

    folds = get_folds(df_selected, FOLD_LENGTH, FOLD_STRIDE)

    for fold_id, fold in enumerate(folds):

        # 1 - Division train / test
        (fold_train, fold_test) = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)

        X_train, y_train = get_X_y(fold_train, N_TRAIN, INPUT_LENGTH, OUTPUT_LENGTH)
        X_test,  y_test  = get_X_y(fold_test,  N_TEST,  INPUT_LENGTH, OUTPUT_LENGTH)

        # [FIX-5] Normalisation de y — fitté sur y_train uniquement
        target_scaler   = TargetScaler()
        y_train_scaled  = target_scaler.fit_transform(y_train)
        y_test_scaled   = target_scaler.transform(y_test)
        y_test_original = target_scaler.inverse_transform(y_test_scaled)

        # 2a - Baseline évalué sur les valeurs originales (non normalisées)
        baseline_model = init_baseline()
        mae_baseline   = baseline_model.evaluate(X_test, y_test_original, verbose=0)[1]
        list_of_mae_baseline_model.append(mae_baseline)
        print("-" * 50)
        print(f"MAE baseline fold n°{fold_id} = {round(mae_baseline, 2)}")

        # 2b - LSTM : entraînement sur y normalisé
        model = init_model(X_train, y_train_scaled)

        es = EarlyStopping(
            # [FIX-3] monitor=val_loss + patience=5 (vs patience=2 + val_mae avant)
            # val_loss est plus stable ; patience=2 stoppait trop tôt sur les folds volatils
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train, y_train_scaled,
            validation_split=0.1,
            shuffle=False,
            batch_size=16,
            epochs=100,
            callbacks=[es],
            verbose=0
        )

        # [FIX-5] Dénormalisation avant calcul de la MAE
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred        = target_scaler.inverse_transform(y_pred_scaled)
        mae_lstm      = float(np.mean(np.abs(y_pred - y_test_original)))
        list_of_mae_recurrent_model.append(mae_lstm)

        print(f"MAE LSTM fold n°{fold_id} = {round(mae_lstm, 2)}")
        print(f"Improvement over baseline: {round((1 - (mae_lstm / mae_baseline)) * 100, 2)} %\n")

    return list_of_mae_baseline_model, list_of_mae_recurrent_model


mae_baselines, mae_lstms = cross_validate_baseline_and_lstm()
