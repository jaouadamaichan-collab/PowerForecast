import pandas as pd
from power_forecast.logic.get_data.build_dataframe import build_common_dataframe, add_features_RNN
pd.set_option('display.max_columns', None)

# Chargement du DataFrame de features à partir d'un CSV multi-pays.
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

df = df_rnn

from typing import Dict, List, Tuple, Sequence
from sklearn.preprocessing import RobustScaler
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

def add_rolling_zscore(df: pd.DataFrame, target: str, window: int = 168) -> pd.DataFrame:
    """
    Ajoute une feature de z-score glissant sur la colonne cible.

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
# [FIX-7] alpha=0.05 → réduit le nombre de features (~15-20)       #
# Un alpha plus élevé force plus de coefficients à zéro, réduisant  #
# la dimensionnalité et facilitant l'apprentissage du LSTM.         #
# ================================================================= #

def lasso_feature_selection(df: pd.DataFrame, target: str,
                             alpha: float = 0.05) -> List[str]:
    """
    Sélectionne les features pertinentes via régression Lasso.

    [FIX-7] alpha passé de 0.01 à 0.05 pour réduire le nombre de features
    de ~47 à ~15-20. Moins de features = moins de bruit = LSTM plus facile
    à entraîner avec un nombre limité de séquences.

    Args:
        df (pd.DataFrame): DataFrame complet avec toutes les features.
        target (str): Nom de la colonne cible.
        alpha (float): Force de régularisation Lasso (défaut : 0.05).

    Returns:
        List[str]: Liste des noms de features dont le coefficient Lasso != 0.
    """
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


# [FIX-7] alpha=0.05 au lieu de 0.01
selected_features = lasso_feature_selection(df, target='FRA', alpha=0.05)

if 'FRA_zscore' not in selected_features:
    selected_features.append('FRA_zscore')
    print("[FIX-4] 'FRA_zscore' forcée dans les features (non sélectionnée par Lasso).")

df_selected = df[selected_features + ['FRA']]
print(f"\nShape après feature selection : {df_selected.shape}")

def clean_target_outliers(df: pd.DataFrame, target: str,
                          lower_pct: float = 0.02,
                          upper_pct: float = 0.98) -> pd.DataFrame:
    df = df.copy()  # évite le SettingWithCopyWarning
    low  = df[target].quantile(lower_pct)
    high = df[target].quantile(upper_pct)
    n_outliers = ((df[target] < low) | (df[target] > high)).sum()
    print(f"Clipping {n_outliers} outliers : [{low:.1f}, {high:.1f}]")
    df[target] = df[target].clip(low, high)
    return df

# --------------------------------------------------- #
# Configuration globale du jeu de données             #
# --------------------------------------------------- #

TARGET          = 'FRA'
N_FEATURES      = df_selected.shape[1]

FOLD_LENGTH      = 24 * 365 * 4
FOLD_STRIDE      = 24 * 182
TRAIN_TEST_RATIO = 0.8

INPUT_LENGTH    = 24 * 7
OUTPUT_LENGTH   = 1
SEQUENCE_STRIDE = 6
DAY_AHEAD_GAP   = 24    # gap cohérent entre X et y dans toutes les fonctions

print(f"N_FEATURES = {N_FEATURES} | INPUT_LENGTH = {INPUT_LENGTH}h = {INPUT_LENGTH//24} jours")
print(f"FOLD_LENGTH = {FOLD_LENGTH}h = {FOLD_LENGTH//24/365:.1f} ans")

def get_folds(
    df: pd.DataFrame,
    fold_length: int,
    fold_stride: int) -> List[pd.DataFrame]:
    """
    Parcourt un DataFrame de série temporelle pour en extraire des folds de longueur fixe.

    Args:
        df (pd.DataFrame): Le DataFrame complet de la série temporelle.
        fold_length (int): Nombre de lignes dans chaque fold.
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

    Args:
        fold (pd.DataFrame): Un fold de pas de temps.
        train_test_ratio (float): Proportion de lignes allouées à l'entraînement.
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

class TargetScaler:
    """
    Normalise et dénormalise la target (y) indépendamment des features (X).
    """

    def __init__(self):
        self.scaler = RobustScaler()  # ← remplace StandardScaler
        self._is_fitted = False

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        shape = y.shape
        y_flat = y.reshape(-1, 1)
        y_scaled = self.scaler.fit_transform(y_flat)
        self._is_fitted = True
        return y_scaled.reshape(shape)

    def transform(self, y: np.ndarray) -> np.ndarray:
        assert self._is_fitted, "Appeler fit_transform avant transform."
        shape = y.shape
        return self.scaler.transform(y.reshape(-1, 1)).reshape(shape)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        assert self._is_fitted, "Appeler fit_transform avant inverse_transform."
        shape = y_scaled.shape
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(shape)

# ================================================================= #
# [FIX-6] get_X_y_strides AVEC GAP COHÉRENT                        #
# ================================================================= #
# Remplace get_X_y (aléatoire) en cross-validation.
# - Extrait ~1300 séquences au lieu de 504 sur un fold standard
# - Gap DAY_AHEAD_GAP=24h appliqué de manière cohérente entre X et y
# - Utilisé partout : évaluation unitaire ET cross-validation

def get_X_y_strides(
    fold: pd.DataFrame,
    input_length: int,
    output_length: int,
    sequence_stride: int,
    gap: int = DAY_AHEAD_GAP
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit un jeu de données en faisant glisser une fenêtre à pas fixe sur le fold.

    [FIX-6] Utilisé en cross-validation pour extraire ~1300 séquences au lieu de 504.
    [FIX-6] Gap DAY_AHEAD_GAP appliqué de manière cohérente (corrige l'incohérence
            entre get_Xi_yi (gap=24) et l'ancienne get_X_y_strides (gap=0)).

    Args:
        fold (pd.DataFrame): Un seul fold de la série temporelle.
        input_length (int): Nombre de pas de temps dans chaque fenêtre X_i.
        output_length (int): Nombre de pas de temps dans chaque cible y_i.
        sequence_stride (int): Nombre de pas de temps entre deux séquences consécutives.
        gap (int): Nombre de pas de temps entre la fin de X et le début de y (défaut : 24h).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y)
    """
    X, y = [], []
    for i in range(0, len(fold), sequence_stride):
        end = i + input_length + gap + output_length
        if end > len(fold):
            break
        X_i = fold.iloc[i:i + input_length].values
        y_i = fold.iloc[i + input_length + gap:
                        i + input_length + gap + output_length][[TARGET]].values
        X.append(X_i)
        y.append(y_i)
    return np.array(X), np.array(y)


X_train, y_train = get_X_y_strides(fold_train, INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)
X_test,  y_test  = get_X_y_strides(fold_test,  INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

# [FIX-5] Normalisation de y — fitté sur train uniquement
target_scaler = TargetScaler()
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled  = target_scaler.transform(y_test)

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train_scaled.shape}")
print(f"X_test  shape : {X_test.shape}")
print(f"y_test  shape : {y_test_scaled.shape}")

# ================================================================= #
# [FIX-8] ARCHITECTURE LSTM AVEC DENSE DE PROJECTION               #
# ================================================================= #

def init_model(X_train: np.ndarray, y_train: np.ndarray) -> tf.keras.Model:
    """
    Construit et compile le modèle LSTM pour la prévision de série temporelle.

    Architecture :
        1. Couche de normalisation — standardise chaque feature.
        2. Dense(16, relu) — [FIX-8] projette les 47→~15 features dans un espace
           plus compact avant le LSTM. Réduit le bruit et facilite l'apprentissage.
        3. Couche LSTM (64 unités, tanh) — capture les dépendances temporelles.
        4. Couche Dense de sortie — projection linéaire sur output_length prédictions.

    Args:
        X_train (np.ndarray): Entrées d'entraînement (n_samples, input_length, n_features).
        y_train (np.ndarray): Cibles d'entraînement normalisées (n_samples, output_length, 1).

    Returns:
        tf.keras.Model: Modèle Keras compilé, prêt pour l'entraînement.
    """
    normalizer = Normalization()
    normalizer.adapt(X_train)

    model = models.Sequential([
        Input(shape=X_train[0].shape),
        normalizer,
        layers.LSTM(
            64,
            activation='tanh',
            return_sequences=True,          # return_sequences=True pour empiler
        ),
        layers.Dropout(0.2),
        layers.LSTM(
            32,
            activation='tanh',
            return_sequences=False,
        ),
        layers.Dropout(0.2),                # Dropout avant la sortie
        layers.Dense(y_train.shape[1], activation='linear')
    ])

    model.compile(
        loss='mse',
        optimizer=optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        metrics=["mae"]
    )

    return model

"""
    Entraîne le modèle LSTM avec arrêt anticipé (early stopping) sur la perte de validation.

    [FIX-3] patience=20 (vs 10 avant) pour laisser plus de temps au modèle de converger,
    notamment sur les folds avec peu de variance.
    """
def fit_model(model, X_train, y_train_scaled, verbose=1):
    # Prendre 10% du milieu comme validation au lieu de la fin
    n = len(X_train)
    val_start = int(n * 0.45)
    val_end   = int(n * 0.55)

    X_val = X_train[val_start:val_end]
    y_val = y_train_scaled[val_start:val_end]

    X_tr = np.concatenate([X_train[:val_start], X_train[val_end:]])
    y_tr = np.concatenate([y_train_scaled[:val_start], y_train_scaled[val_end:]])

    es = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),  # segment central, plus représentatif
        shuffle=False,
        batch_size=64,
        epochs=200,
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

# ====================================
# 3 - Évaluation sur le test set
# ====================================
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred        = target_scaler.inverse_transform(y_pred_scaled).squeeze()
y_test_orig   = target_scaler.inverse_transform(y_test_scaled).squeeze()
mae_lstm      = float(np.mean(np.abs(y_pred - y_test_orig)))

print(f"The LSTM MAE on the test set is equal to {round(mae_lstm, 2)}")

y_pred_scaled = model.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test_original = target_scaler.inverse_transform(y_test_scaled).squeeze()  # ← squeeze
y_pred = y_pred.squeeze()                                                    # ← squeeze

mae_lstm = np.mean(np.abs(y_pred - y_test_original))
print(f"The LSTM MAE on the test set is equal to {round(mae_lstm, 2)}")

print("y_train avant scaling — mean :", y_train.mean(), "| std :", y_train.std())
print("y_train_scaled — mean :", y_train_scaled.mean(), "| std :", y_train_scaled.std())

print(f"y_train — mean: {y_train.mean():.1f} | std: {y_train.std():.1f} | min: {y_train.min():.1f} | max: {y_train.max():.1f}")
print(f"y_test  — mean: {y_test.mean():.1f}  | std: {y_test.std():.1f}  | min: {y_test.min():.1f}  | max: {y_test.max():.1f}")

# Et surtout : est-ce que le modèle prédit des valeurs sensées ?
y_pred_scaled = model.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
print(f"\ny_pred  — mean: {y_pred.mean():.1f}  | std: {y_pred.std():.1f}  | min: {y_pred.min():.1f}  | max: {y_pred.max():.1f}")
print(f"y_test  — mean: {y_test.mean():.1f}  | std: {y_test.std():.1f}")

# Inspecte les valeurs aberrantes
print("Valeurs y_train < 0 :")
print(np.where(y_train < 0))
print(y_train[y_train < 0])

print("\nValeurs y_train > 300 (potentielles anomalies) :")
print(y_train[y_train > 300])

# Où dans la série temporelle ?
bad_idx = np.where(y_train.flatten() < 0)[0]
print(f"\nIndices séquences avec y < 0 : {bad_idx}")
print(f"Valeurs correspondantes dans df : {df_selected['FRA'].iloc[bad_idx]}")

y_val_pred = model.predict(X_train[-len(X_train)//10:])
print(f"Meilleure val_loss annoncée : {min(history.history['val_loss']):.4f}")
print(f"Epoch du meilleur           : {np.argmin(history.history['val_loss']) + 1}")

# ================================================================= #
# [FIX-9] BASELINE SAME-HOUR-7-DAYS-AGO                            #
# ================================================================= #
# L'ancienne baseline "dernière valeur observée" (x[:, -1]) était trop
# facile à battre car elle ne respectait pas la saisonnalité hebdomadaire.
# La nouvelle baseline prend la valeur de la même heure 7 jours avant,
# ce qui est le prédicteur naïf standard en prévision de consommation.
#
# Index dans X : la séquence X couvre [t-INPUT_LENGTH, t-1].
# La cible y est à t + DAY_AHEAD_GAP (24h après la fin de X).
# La même heure 7 jours avant la cible = t + 24 - 7*24 = t - 144.
# Dans X (0-indexé depuis le début de la fenêtre) :
#   index = INPUT_LENGTH - (7*24 - DAY_AHEAD_GAP) = 168 - (168 - 24) = 24
# → on prend X[:, 24, fra_idx] comme prédiction baseline.

fra_idx = list(df_selected.columns).index('FRA')

# Offset dans la fenêtre X correspondant à "même heure, 7 jours avant la cible"
SAME_HOUR_7D_IDX = INPUT_LENGTH - (7 * 24 - DAY_AHEAD_GAP)  # = 24


def init_baseline(fra_idx: int, same_hour_idx: int) -> tf.keras.Model:
    """
    Baseline "même heure, 7 jours avant" — prédicteur naïf saisonnier.

    [FIX-9] Remplace l'ancienne baseline "dernière valeur observée" (x[:, -1, 1]).
    La saisonnalité hebdomadaire est le principal pattern de la consommation
    électrique ; cette baseline le capture directement.

    La valeur lue est à l'index `same_hour_idx` dans la dimension temporelle
    de X, correspondant à la même heure exactement 7 jours avant la cible.

    Args:
        fra_idx (int): Index de la colonne FRA dans la dimension features de X.
        same_hour_idx (int): Index temporel dans X de la valeur "même heure -7j".

    Returns:
        tf.keras.Model: Modèle Keras compilé retournant la valeur saisonnière.
    """
    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x[:, same_hour_idx, fra_idx, None]))

    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model


baseline_model = init_baseline(fra_idx, SAME_HOUR_7D_IDX)
baseline_score = baseline_model.evaluate(X_test, y_test_original)

print(f"\n--- Résultats fold 0 ---")
print(f"Baseline MAE (même heure -7j) : {round(baseline_score[1], 2)}")
print(f"LSTM MAE                      : {round(mae_lstm, 2)}")
print(f"Amélioration LSTM vs baseline : {round((1 - (mae_lstm / baseline_score[1])) * 100, 2)} %")

# ================================================================= #
# [FIX-10] SANITY CHECK : MAE TRAIN VS TEST                        #
# ================================================================= #
# Permet de distinguer deux cas d'échec :
#   - MAE train élevée (~= MAE test) → sous-apprentissage (modèle n'apprend rien)
#   - MAE train faible, MAE test élevée → surapprentissage (modèle ne généralise pas)
# Si MAE train >> baseline sur train → architecture ou données problématiques.

y_pred_train_scaled = model.predict(X_train, verbose=0)
y_pred_train        = target_scaler.inverse_transform(y_pred_train_scaled).squeeze()
y_train_orig        = target_scaler.inverse_transform(y_train_scaled).squeeze()
mae_train           = float(np.mean(np.abs(y_pred_train - y_train_orig)))

mae_baseline = baseline_model.evaluate(X_test, y_test_orig, verbose=0)[1]
# mae_baseline_train = mae_baseline[1]

print(f"\n--- [FIX-10] Sanity Check ---")
print(f"MAE LSTM     sur TRAIN : {round(mae_train, 2)}")
print(f"MAE LSTM     sur TEST  : {round(mae_lstm, 2)}")
print(f"MAE Baseline sur TRAIN : {round(mae_baseline, 2)}")
print(f"MAE Baseline sur TEST  : {round(baseline_score[1], 2)}")
print()
if mae_train > baseline_score[1] * 0.9:
    print("⚠️  SOUS-APPRENTISSAGE détecté : MAE train proche ou > baseline.")
    print("   → Essayer : plus d'epochs, learning_rate plus élevé, moins de régularisation.")
elif mae_lstm > mae_train * 2:
    print("⚠️  SURAPPRENTISSAGE détecté : MAE test >> MAE train.")
    print("   → Essayer : plus de Dropout, plus de données, régularisation L2 plus forte.")
else:
    print("✅  Pas de problème majeur détecté (train/test cohérents).")


print(f'\nN_FEATURES       = {N_FEATURES}')
print(f'FOLD_LENGTH      = {FOLD_LENGTH}')
print(f'FOLD_STRIDE      = {FOLD_STRIDE}')
print(f'TRAIN_TEST_RATIO = {TRAIN_TEST_RATIO}')
print(f'INPUT_LENGTH     = {INPUT_LENGTH}')
print(f'OUTPUT_LENGTH    = {OUTPUT_LENGTH}')
print(f'SEQUENCE_STRIDE  = {SEQUENCE_STRIDE}')
print(f'DAY_AHEAD_GAP    = {DAY_AHEAD_GAP}')
print(f'SAME_HOUR_7D_IDX = {SAME_HOUR_7D_IDX}')
print(f'fra_idx          = {fra_idx}')


# ================================================================= #
# CROSS-VALIDATION                                                   #
# [FIX-6] get_X_y_strides au lieu de get_X_y (aléatoire)           #
# [FIX-9] Baseline same-hour-7-days-ago                             #
# [FIX-10] Sanity check MAE train par fold                          #
# ================================================================= #

def cross_validate_baseline_and_lstm():
    """
    Effectue une validation croisée temporelle en comparant le baseline et le modèle LSTM.

    Pour chaque fold :
        1. Division en train/test.
        2. [FIX-6] Génération des séquences par strides (cohérent, ~1300 séq. vs 504).
        3. Normalisation de y_train ; transformation (sans re-fit) de y_test. [FIX-5]
        4. Évaluation du baseline naïf same-hour-7-days-ago. [FIX-9]
        5. Entraînement LSTM avec early stopping + gradient clipping.
        6. Dénormalisation des prédictions avant calcul de la MAE.
        7. [FIX-10] Sanity check MAE train pour diagnostiquer sur/sous-apprentissage.

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

        # [FIX-6] get_X_y_strides au lieu de get_X_y (aléatoire)
        # ~1300 séquences extraites systématiquement vs 504 aléatoires
        X_train, y_train = get_X_y_strides(fold_train, INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)
        X_test,  y_test  = get_X_y_strides(fold_test,  INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

        # [FIX-5] Normalisation de y — fitté sur y_train uniquement
        target_scaler   = TargetScaler()
        y_train_scaled  = target_scaler.fit_transform(y_train)
        y_test_scaled   = target_scaler.transform(y_test)
        y_test_orig     = target_scaler.inverse_transform(y_test_scaled).squeeze()

        # 2a - [FIX-9] Baseline same-hour-7-days-ago
        baseline_model = init_baseline(fra_idx, SAME_HOUR_7D_IDX)
        mae_baseline = baseline_model.evaluate(X_test, y_test_orig, verbose=0)[1]
        list_of_mae_baseline_model.append(mae_baseline)

        print("-" * 50)
        print(f"MAE baseline fold n°{fold_id + 1} = {round(mae_baseline, 2)}")

        # 2b - LSTM
        model = init_model(X_train, y_train_scaled)

        es = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=20,   # [FIX-3] patience augmentée
            restore_best_weights=True
        )

        n = len(X_train)
        val_s, val_e = int(n * 0.45), int(n * 0.55)
        X_val_cv = X_train[val_s:val_e]
        y_val_cv  = y_train_scaled[val_s:val_e]
        X_tr_cv   = np.concatenate([X_train[:val_s], X_train[val_e:]])
        y_tr_cv   = np.concatenate([y_train_scaled[:val_s], y_train_scaled[val_e:]])

        model.fit(
            X_tr_cv, y_tr_cv,
            validation_data=(X_val_cv, y_val_cv),
            shuffle=False,
            batch_size=64,
            epochs=200,
            callbacks=[es],
            verbose=0
        )

        # [FIX-5] Dénormalisation avant calcul de la MAE
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred        = target_scaler.inverse_transform(y_pred_scaled).squeeze()
        mae_lstm      = float(np.mean(np.abs(y_pred - y_test_orig)))
        list_of_mae_recurrent_model.append(mae_lstm)

        print(f"MAE LSTM fold n°{fold_id + 1} = {round(mae_lstm, 2)}")
        print(f"Improvement over baseline: {round((1 - (mae_lstm / mae_baseline)) * 100, 2)} %")

        # [FIX-10] Sanity check : MAE train pour détecter sur/sous-apprentissage
        y_pred_train_scaled = model.predict(X_train, verbose=0)
        y_pred_train        = target_scaler.inverse_transform(y_pred_train_scaled).squeeze()
        y_train_orig        = target_scaler.inverse_transform(y_train_scaled).squeeze()
        mae_train           = float(np.mean(np.abs(y_pred_train - y_train_orig)))



        print(f"  [Sanity] MAE TRAIN = {round(mae_train, 2)} | MAE TEST = {round(mae_lstm, 2)}", end="  ")
        if mae_train > mae_baseline * 0.9:
            print("⚠️  SOUS-APPRENTISSAGE")
        elif mae_lstm > mae_train * 2:
            print("⚠️  SURAPPRENTISSAGE")
        else:
            print("✅")

    return list_of_mae_baseline_model, list_of_mae_recurrent_model


mae_baselines, mae_lstms = cross_validate_baseline_and_lstm()

# ================================================ #
# Résumé final de la cross-validation              #
# ================================================ #
print("\n" + "=" * 50)
print("RÉSUMÉ CROSS-VALIDATION")
print("=" * 50)
for i, (b, l) in enumerate(zip(mae_baselines, mae_lstms)):
    improvement = round((1 - l / b) * 100, 2)
    status = "✅" if improvement > 0 else "❌"
    print(f"Fold {i+1:2d} | Baseline: {b:7.2f} | LSTM: {l:7.2f} | Δ: {improvement:+.1f}% {status}")

mean_b = np.mean(mae_baselines)
mean_l = np.mean(mae_lstms)
mean_imp = round((1 - mean_l / mean_b) * 100, 2)
print("-" * 50)
print(f"MOYENNE | Baseline: {mean_b:7.2f} | LSTM: {mean_l:7.2f} | Δ: {mean_imp:+.1f}%")
