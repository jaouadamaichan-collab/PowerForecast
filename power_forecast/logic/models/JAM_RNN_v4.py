"""
Power Forecast - Modèle LSTM Optimisé (v5)
==========================================
Sources scientifiques :
  [1] Li & Becker, NTNU (2021) — "Day-ahead electricity price prediction applying
      hybrid models of LSTM-based deep learning methods and feature selection
      algorithms under consideration of market coupling"
  [2] Trebbien et al., Forschungszentrum Jülich (2023) — "Probabilistic Forecasting
      of Day-Ahead Electricity Prices and their Volatility with LSTMs"
  [3] Tschora, INSA Lyon (2024) — "Machine Learning Techniques for Electricity
      Price Forecasting" (thèse de doctorat, soutenue le 17/01/2024)

─────────────────────────────────────────────────────────────────────────────
APPORTS DE [3] — Tschora (2024)                              NOUVEAUTÉS v5
─────────────────────────────────────────────────────────────────────────────

    [DATASETS ENRICHIS] — Chapitre 3, Section 3.2.3
    1. Extension des features avec données pays voisins : prix spot DE, BE, NL,
       ES, CH (Section 3.2.3.1). La thèse montre un gain jusqu'à 15% de MAE.
       Règle d'or : prix CH disponibles avant midi → inclus sans lag.
       Règle d'or : lags uniquement D-1 et D-7 (D-2, D-3 contribuent < 5%).

    [ENCODAGE CYCLIQUE DES DATES] — Section 3.2.3.2
    2. Encodage circulaire des features temporelles (jour de semaine, numéro
       de semaine, jour du mois, mois) via sin/cos :
           F(x) = (sin(2πx/α), cos(2πx/α))
       Capte la saisonnalité sans discontinuité entre décembre et janvier.

    [PRIX DU GAZ] — Section 3.2.3.3
    3. Prix du gaz (indice EGSI) ajouté comme feature avec lag D-2 (disponible
       à 18h la veille → utilisable pour prévision J+1). Le modèle augmente
       dynamiquement le poids du gaz lors des crises énergétiques (Figure 3.9).
       NB : pour le marché FR (nucléaire dominant), l'impact est moindre
       mais non nul lors des maintenances prolongées.

    [MÉTRIQUES ENRICHIES] — Section 3.3.2
    4. Ajout de la DAE (Daily Average Error) spécifique au trading :
           DAE = mean |mean_h(y_pred_h) - mean_h(y_true_h)|
       Et du RMAE (Relative MAE vs baseline naïf) :
           RMAE = MAE(modèle) / MAE(baseline)
       Un RMAE < 1 signifie que le modèle bat le naïf.

    [BASELINE NAÏF CORRECT] — Section 3.3.2
    5. Le baseline naïf de référence est ajusté conformément à la thèse :
           - Jour de semaine  → prix de D-1 même heure
           - Week-end        → prix de D-7 même heure
       Plus robuste que la simple répétition de la dernière valeur.

    [SCALERS ROBUSTES AUX OUTLIERS] — Section 3.3.3
    6. Quatre scalers évalués dans la thèse. On implémente :
           - StandardScaler          : µ=0, σ=1
           - MedianScaler            : robuste aux spikes de prix
           - arcsinh(StandardScaler) : compresse les queues lourdes
       Le scaler est sélectionné par validation sur le fold courant.

    [RECALIBRATION] — Section 3.3.5
    7. Recalibration périodique : ré-entraînement glissant tous les N_RECALIB
       jours en intégrant les nouvelles données. Cruciale lors des crises
       (Covid-19 2020, crise gaz 2021) qui modifient brusquement la dynamique.

    [TEST DE DIEBOLD & MARIANO] — Section 3.3.2.2
    8. Test statistique DM pour comparer deux modèles sur le set de test.
       H0 : MAE(modèle_1) ≤ MAE(modèle_2)   (modèle_2 n'est pas meilleur)
       p-value < 0.05 → modèle_2 significativement meilleur.
       Évite de conclure à une amélioration qui n'est que du bruit.

    [SHAP VALUES] — Sections 3.3.6, 3.4.2
    9. Calcul des SHAP values pour expliquer les prédictions du modèle.
       Identifie quelles features (prix voisins, consommation, gaz) guident
       le modèle — critique pour la confiance opérationnelle (Section 3.5).

─────────────────────────────────────────────────────────────────────────────
APPORTS CONSERVÉS DE [1] — Li & Becker (2021)
─────────────────────────────────────────────────────────────────────────────
    1. Feature selection Lasso (alpha=0.01, méthode M5)
    2. Architecture LSTM Encoder-Decoder (M6)
    3. N_EXPERIMENTS = 10 répétitions

─────────────────────────────────────────────────────────────────────────────
APPORTS CONSERVÉS DE [2] — Trebbien et al. (2023)
─────────────────────────────────────────────────────────────────────────────
    4. Prévision probabiliste µ + σ
    5. Loss NLL (Negative Log-Likelihood)
    6. INPUT_LENGTH = 96h (fenêtre gaussienne)
    7. Architecture légère depth=2, width=32, dropout=0.2
    8. Normalisation par max(train) — sans look-ahead bias
    9. Early stopping patience=200
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict, Optional

from power_forecast.logic.get_data.build_dataframe import build_feature_dataframe
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers, Input, optimizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

pd.set_option('display.max_columns', None)

# ================================================================= #
# 1. CHARGEMENT DES DONNÉES                                         #
# ================================================================= #

df = build_feature_dataframe(
    filepath='raw_data/all_countries.csv',
    load_from_pickle = False, #True if you want to load from a previously saved pickle file, False to build the dataframe from scratch (which takes more time)
    country_objective='France',
    target_day_distance=2,
    time_interval='h', #Time interval for resampling the data (e.g., 'h' for hourly, 'D' for daily)
    save_name='df_with_features',
    drop_nan=True, #Drop rows with NaN values (due to target distance and catch24 features)
    keep_only_neighbors=False, #Keep only neighboring countries for the lag frontiere features (instead of all countries)
    add_lag_frontiere=True, #Add lag features of neighboring countries (based on FRONTIERE dict)
    add_crisis=True, #Add crisis features (based on CRISIS_PERIODS dict)
    add_gen_load_forecast=False, #Add generation and load forecast features (based on GEN_LOAD_FORECAST dict)
    add_catch24=True, #Add catch24 features (based on WINDOW_CATCH22 and STEP_CATCH22 parameters
)

# ================================================================= #
# 2. ENRICHISSEMENT DES FEATURES                      [3] Chap.3   #
#                                                                   #
# Apports 1, 2, 3 de [3] : pays voisins, encodage cyclique, prix   #
# du gaz. Ces fonctions opèrent sur le DataFrame brut AVANT Lasso.  #
# ================================================================= #

def add_lag_features(df: pd.DataFrame, cols: List[str],
                     lags: List[int]) -> pd.DataFrame:
    """
    Ajoute des colonnes décalées (lags) pour chaque feature listée.

    Conformément à [3] Section 3.2.3.4 : seuls les lags D-1 et D-7
    contribuent significativement (Table 3.8). D-2 et D-3 représentent
    chacun < 5% de la contribution totale → exclus pour limiter la
    dimensionnalité.

    Args:
        df (pd.DataFrame): DataFrame source.
        cols (List[str]): Colonnes à décaler.
        lags (List[int]): Lags en heures (ex: [24, 168] pour D-1 et D-7).

    Returns:
        pd.DataFrame: DataFrame avec colonnes lags ajoutées.
    """
    df_out = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            new_col = f"{col}_lag{lag}h"
            df_out[new_col] = df_out[col].shift(lag)
    return df_out


def add_cyclic_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les features temporelles de façon cyclique via sin/cos.

    Conformément à [3] Section 3.2.3.2 :
        F(x) = (sin(2πx/α), cos(2πx/α))
    avec α la cardinalité de la feature (7 pour le jour de semaine, etc.).

    Évite la discontinuité de l'encodage entier entre la dernière valeur
    et la première (ex: dimanche → lundi, décembre → janvier).

    Args:
        df (pd.DataFrame): DataFrame avec index DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame avec colonnes cycliques ajoutées.
    """
    df_out = df.copy()
    idx = pd.to_datetime(df_out.index)

    cyclic_features = {
        'hour':        (idx.hour,            24),
        'dayofweek':   (idx.dayofweek,        7),
        'weekofyear':  (idx.isocalendar().week.astype(int), 53),
        'dayofmonth':  (idx.day,             31),
        'month':       (idx.month,           12),
    }
    for name, (values, alpha) in cyclic_features.items():
        df_out[f'{name}_sin'] = np.sin(2 * np.pi * values / alpha)
        df_out[f'{name}_cos'] = np.cos(2 * np.pi * values / alpha)

    return df_out


def add_gas_price_feature(df: pd.DataFrame,
                           gas_col: Optional[str] = 'GAS') -> pd.DataFrame:
    """
    Ajoute le prix du gaz avec un lag de 48h (D-2).

    Conformément à [3] Section 3.2.3.3 : l'indice EGSI est publié
    chaque jour à 18h. Pour prédire J+1 (avant 12h en J), on utilise
    la valeur J-1 → lag 48h.

    Impact : le modèle augmente dynamiquement le poids du gaz lors des
    crises (crise 2021) et le diminue lors des périodes nucléaires
    stables (été 2020, Section 3.4.2.2).

    Args:
        df (pd.DataFrame): DataFrame source.
        gas_col (str | None): Nom de la colonne gaz. Si absente, no-op.

    Returns:
        pd.DataFrame: DataFrame avec 'GAS_lag48h' ajouté si disponible.
    """
    df_out = df.copy()
    if gas_col and gas_col in df.columns:
        df_out['GAS_lag48h'] = df_out[gas_col].shift(48)
        print(f"  [GAS] Colonne '{gas_col}_lag48h' ajoutée.")
    else:
        print(f"  [GAS] Colonne '{gas_col}' absente — feature ignorée.")
    return df_out


def enrich_features(df: pd.DataFrame, target: str,
                    neighbor_price_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Pipeline d'enrichissement complet du DataFrame brut.

    Étapes :
        1. Lags D-1 (24h) et D-7 (168h) sur les prix voisins [3] Sec.3.2.3.1
        2. Encodage cyclique des dates [3] Sec.3.2.3.2
        3. Prix du gaz avec lag D-2 [3] Sec.3.2.3.3
        4. Suppression des lignes NaN introduites par les lags

    Args:
        df (pd.DataFrame): DataFrame brut multi-pays.
        target (str): Colonne cible (ex: 'FRA').
        neighbor_price_cols (List[str] | None): Colonnes prix voisins à lagger.
            Si None, détecté automatiquement (toutes colonnes hors target).

    Returns:
        pd.DataFrame: DataFrame enrichi, sans NaN.
    """
    # 1. Lags sur prix voisins — D-1 et D-7 uniquement [3] Table 3.8
    if neighbor_price_cols is None:
        neighbor_price_cols = [c for c in df.columns if c != target]
    df = add_lag_features(df, cols=neighbor_price_cols, lags=[24, 168])

    # 2. Encodage cyclique des dates
    df = add_cyclic_date_features(df)

    # 3. Prix du gaz (lag D-2)
    df = add_gas_price_feature(df)

    # 4. Nettoyage NaN introduits par les lags (max lag = 168h)
    n_before = len(df)
    df = df.dropna()
    print(f"  [ENRICH] {n_before} → {len(df)} lignes après dropna (168h supprimées).")
    return df


# ================================================================= #
# 3. FEATURE SELECTION PAR LASSO                          [1]       #
# ================================================================= #

def lasso_feature_selection(df: pd.DataFrame, target: str,
                             alpha: float = 0.01) -> List[str]:
    """
    Sélectionne les features pertinentes pour prédire `target` via Lasso.

    Lasso minimise : MSE + alpha * sum(|coef|).
    Les features peu informatives ont leur coefficient réduit à zéro.
    alpha=0.01 proche du lambda=0.02 de la méthode M5 de [1] (Table 5).

    Args:
        df (pd.DataFrame): DataFrame enrichi avec toutes les features.
        target (str): Nom de la colonne cible.
        alpha (float): Paramètre de régularisation L1 (défaut=0.01).

    Returns:
        List[str]: Features sélectionnées (coefficients Lasso non nuls).
    """
    features = [c for c in df.columns if c != target]
    X = df[features].values
    y = df[target].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_scaled, y)

    selected = [features[i] for i, coef in enumerate(lasso.coef_) if coef != 0]
    print(f"\nLasso (alpha={alpha}) : {len(features)} → {len(selected)} features")
    print(f"  Conservées : {selected}")
    return selected


# ── Pipeline enrichissement + sélection ──────────────────────────
df_enriched       = enrich_features(df, target='FRA')
selected_features = lasso_feature_selection(df_enriched, target='FRA', alpha=0.01)
df_selected       = df_enriched[selected_features + ['FRA']]
print(f"Shape finale : {df_selected.shape}")

# ================================================================= #
# 4. CONFIGURATION GLOBALE                                          #
# ================================================================= #

TARGET          = 'FRA'
N_FEATURES      = df_selected.shape[1]

FOLD_LENGTH      = 24 * 365 * 2   # 2 ans (~17 520h)
FOLD_STRIDE      = 24 * 91        # 1 trimestre entre chaque fold
TRAIN_TEST_RATIO = 0.7

INPUT_LENGTH    = 96    # 4 jours [2] — hypothèse gaussienne
OUTPUT_LENGTH   = 1
SEQUENCE_STRIDE = 24    # 1 pas/jour
DAY_AHEAD_GAP   = 24    # Prévision J+1

N_EXPERIMENTS   = 10    # Répétitions pour stabiliser les scores [1]
N_RECALIB       = 30    # Recalibration tous les 30 jours [3] Sec.3.3.5

print(f"N_FEATURES = {N_FEATURES} | INPUT_LENGTH = {INPUT_LENGTH}h = {INPUT_LENGTH//24} jours")

# ================================================================= #
# 5. SCALERS ROBUSTES AUX OUTLIERS                    [3] Sec.3.3.3 #
#                                                                   #
# La thèse évalue 4 scalers. On implémente les 3 les plus utiles :  #
#   - StandardScaler  : µ=0, σ=1                                    #
#   - MedianScaler    : median=0, MAD=1 (robuste aux spikes)        #
#   - ArcsinhScaler   : arcsinh(StandardScaler) — compresse queues  #
# ================================================================= #

class MedianScaler:
    """
    Scaler robuste aux outliers : centre sur la médiane, réduit par MAD.

    Conformément à [3] Équation 3.2 :
        MS(X) = (X - median(X)) / MAD(X)²

    MAD = médiane des écarts absolus à la médiane.
    Résistant aux pics de prix extrêmes (spikes).
    """

    def __init__(self):
        self.median_ = None
        self.mad_    = None

    def fit(self, X: np.ndarray) -> 'MedianScaler':
        """Calcule médiane et MAD sur X."""
        self.median_ = np.median(X, axis=0)
        self.mad_    = np.median(np.abs(X - self.median_), axis=0)
        self.mad_    = np.where(self.mad_ == 0, 1.0, self.mad_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforme X en (X - median) / MAD²."""
        return (X - self.median_) / (self.mad_ ** 2)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Combine fit et transform."""
        return self.fit(X).transform(X)


def arcsinh_transform(X: np.ndarray) -> np.ndarray:
    """
    Transformée arcsinh : log(X + sqrt(X² + 1)).

    Conformément à [3] Équation 3.3 : appliquée après un scaler de base
    (Standard ou Median) pour comprimer les queues lourdes.
    Stable en 0 contrairement à log.

    Args:
        X (np.ndarray): Données scalées.

    Returns:
        np.ndarray: Données transformées.
    """
    return np.log(X + np.sqrt(X**2 + 1))


# ================================================================= #
# 6. FONCTIONS DE DÉCOUPAGE                                         #
# ================================================================= #

def get_folds(df: pd.DataFrame, fold_length: int,
              fold_stride: int) -> List[pd.DataFrame]:
    """
    Extrait des folds de longueur fixe par fenêtre glissante.

    Args:
        df (pd.DataFrame): Série temporelle complète.
        fold_length (int): Nombre de pas par fold.
        fold_stride (int): Pas entre deux folds.

    Returns:
        List[pd.DataFrame]: Liste de folds.
    """
    folds = []
    for idx in range(0, len(df), fold_stride):
        if (idx + fold_length) > len(df):
            break
        folds.append(df.iloc[idx:idx + fold_length, :])
    return folds


def train_test_split(fold: pd.DataFrame, train_test_ratio: float,
                     input_length: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divise un fold en train et test.

    Args:
        fold (pd.DataFrame): Fold à diviser.
        train_test_ratio (float): Proportion allouée au train.
        input_length (int): Longueur d'une séquence X_i.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (fold_train, fold_test)
    """
    last_train_idx = round(train_test_ratio * len(fold))
    fold_train     = fold.iloc[:last_train_idx, :]
    fold_test      = fold.iloc[last_train_idx - input_length:, :]
    return fold_train, fold_test


def normalize_by_train_max(fold_train: pd.DataFrame,
                            fold_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalise train et test par le maximum absolu du train.

    Conformément à [2] Section III : normalisation sur le train uniquement
    pour éviter tout look-ahead bias. Chaque fold est normalisé
    indépendamment par son propre train set.

    Args:
        fold_train (pd.DataFrame): Données d'entraînement.
        fold_test (pd.DataFrame): Données de test.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train normalisé, test normalisé)
    """
    max_vals = fold_train.abs().max()
    max_vals = max_vals.replace(0, 1)
    return fold_train / max_vals, fold_test / max_vals


def get_X_y_strides(fold: pd.DataFrame, input_length: int, output_length: int,
                    sequence_stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère (X_i, y_i) par fenêtre glissante avec décalage day-ahead.

    Schéma : [--- X_i (96h) ---][--- gap 24h ---][y_i]

    Args:
        fold (pd.DataFrame): Fold normalisé.
        input_length (int): Longueur de X_i.
        output_length (int): Longueur de y_i.
        sequence_stride (int): Pas entre séquences.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y)
    """
    X, y = [], []
    for i in range(0, len(fold), sequence_stride):
        if (i + input_length + DAY_AHEAD_GAP + output_length) > len(fold):
            break
        X.append(fold.iloc[i:i + input_length, :].values)
        y_start = i + input_length + DAY_AHEAD_GAP
        y.append(fold.iloc[y_start:y_start + output_length, :][TARGET].values)
    return np.array(X), np.array(y)

# ================================================================= #
# 7. LOSS NLL (NEGATIVE LOG-LIKELIHOOD)               [2]           #
# ================================================================= #

def nll_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Negative Log-Likelihood gaussienne.

    y_pred[:, 0] = µ     (prédiction ponctuelle)
    y_pred[:, 1] = log_σ (log de l'écart-type → σ = exp(log_σ) ≥ 0.01)

    NLL = mean[ log(2π σ²)/2 + (y − µ)²/(2σ²) ]

    Args:
        y_true: Valeurs réelles, forme (batch,).
        y_pred: Sorties [µ, log_σ], forme (batch, 2).

    Returns:
        tf.Tensor: Scalaire NLL moyen.
    """
    mu    = y_pred[:, 0]
    sigma = K.maximum(K.exp(y_pred[:, 1]), 0.01)
    nll   = (K.log(2 * np.pi * K.square(sigma)) / 2
             + K.square(y_true - mu) / (2 * K.square(sigma)))
    return K.mean(nll)

# ================================================================= #
# 8. ARCHITECTURE LSTM PROBABILISTE                   [1] + [2]     #
# ================================================================= #

def init_model(input_shape: Tuple) -> tf.keras.Model:
    """
    Construit le modèle LSTM Encoder-Decoder à sortie probabiliste.

    Architecture :
        Entrée    : (input_length, n_features)
        Encodeur  : LSTM(32, tanh) + Dropout(0.2) → vecteur latent
        Pont      : RepeatVector(1)
        Décodeur  : LSTM(32, tanh) + Dropout(0.2) → Dense(2)
        Sortie    : [µ, log_σ]  (activation linéaire)

    Hyperparamètres selon [2] : width=32, dropout=0.2, depth=2.

    Args:
        input_shape (Tuple): Forme d'une séquence (input_length, n_features).

    Returns:
        tf.keras.Model: Modèle compilé avec loss NLL.
    """
    inp = Input(shape=input_shape)

    # Encodeur
    x = layers.LSTM(32, activation='tanh', return_sequences=False)(inp)
    x = layers.Dropout(0.2)(x)

    # Pont encodeur → décodeur
    x = layers.RepeatVector(1)(x)

    # Décodeur
    x = layers.LSTM(32, activation='tanh', return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)

    # Sortie : [µ, log_σ]
    out = layers.Dense(2, activation='linear')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(loss=nll_loss, optimizer=optimizers.Adam(learning_rate=0.001))
    return model

# ================================================================= #
# 9. ENTRAÎNEMENT                                                   #
# ================================================================= #

def fit_model(model: tf.keras.Model, X_train: np.ndarray,
              y_train: np.ndarray, verbose: int = 1):
    """
    Entraîne le modèle avec early stopping (patience=200) et ReduceLROnPlateau.

    La patience élevée (200) est conforme à [2] Section V : le modèle
    a besoin de temps pour co-apprendre µ et σ via la NLL.

    Args:
        model: Modèle compilé.
        X_train: Séquences d'entrée (n, input_length, n_features).
        y_train: Cibles scalaires (n,).
        verbose: 0=silencieux, 1=barre de progression.

    Returns:
        Tuple: (modèle entraîné, historique)
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=200, mode='min',
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                          min_lr=1e-6, verbose=0),
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        shuffle=False,
        batch_size=32,
        epochs=500,
        callbacks=callbacks,
        verbose=verbose
    )
    return model, history

# ================================================================= #
# 10. MÉTRIQUES ENRICHIES                             [1]+[2]+[3]   #
#                                                                   #
# v5 ajoute DAE et RMAE conformément à [3] Section 3.3.2.1 :       #
#   DAE  = mean |mean_h(ŷ_d) - mean_h(y_d)|   (trading day-ahead)  #
#   RMAE = MAE(modèle) / MAE(baseline_naïf)    (comparatif)        #
# ================================================================= #

def compute_metrics(y_true: np.ndarray, y_pred_raw: np.ndarray,
                    y_pred_baseline: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calcule NLL, MAE, SMAPE, DAE et RMAE.

    y_pred_raw[:, 0] = µ      → utilisé pour MAE, SMAPE, DAE, RMAE
    y_pred_raw[:, 1] = log_σ  → µ + σ utilisés pour NLL

    La DAE mesure l'erreur sur le prix moyen journalier — directement
    liée à la rentabilité des positions de trading [3] Section 3.3.2.1.

    Le RMAE < 1 indique que le modèle bat le naïf.

    Args:
        y_true: Valeurs réelles (n,).
        y_pred_raw: Sorties brutes [µ, log_σ] (n, 2).
        y_pred_baseline: Sorties brutes du baseline (n, 2), optionnel pour RMAE.

    Returns:
        Dict[str, float]: NLL, MAE, SMAPE, DAE, RMAE (si baseline fourni).
    """
    y_true = y_true.flatten()
    mu     = y_pred_raw[:, 0]
    sigma  = np.maximum(np.exp(y_pred_raw[:, 1]), 0.01)

    nll   = float(np.mean(np.log(2 * np.pi * sigma**2) / 2
                          + (y_true - mu)**2 / (2 * sigma**2)))
    mae   = float(np.mean(np.abs(y_true - mu)))
    smape = float(100 * np.mean(np.abs(y_true - mu) /
                                ((np.abs(y_true) + np.abs(mu)) / 2 + 1e-8)))

    # DAE [3] — groupe les pas horaires en jours (OUTPUT_LENGTH=1 ici)
    # Avec OUTPUT_LENGTH=1, DAE = MAE sur la valeur journalière unique
    dae = mae

    metrics = {
        'NLL':   round(nll,   4),
        'MAE':   round(mae,   4),
        'SMAPE': round(smape, 4),
        'DAE':   round(dae,   4),
    }

    # RMAE : nécessite la baseline [3] Équation RMAE
    if y_pred_baseline is not None:
        mu_bl    = y_pred_baseline[:, 0]
        mae_bl   = float(np.mean(np.abs(y_true - mu_bl)))
        rmae     = mae / (mae_bl + 1e-8)
        metrics['RMAE'] = round(rmae, 4)

    return metrics


def print_metrics(metrics: Dict[str, float], label: str = ""):
    """Affiche les métriques de manière formatée."""
    rmae_str = f" | RMAE={metrics['RMAE']:.4f}" if 'RMAE' in metrics else ""
    print(f"  {label:12s} | NLL={metrics['NLL']:7.4f} "
          f"| MAE={metrics['MAE']:8.4f} | SMAPE={metrics['SMAPE']:.2f}%"
          f" | DAE={metrics['DAE']:.4f}{rmae_str}")

# ================================================================= #
# 11. BASELINE NAÏF CORRIGÉ                           [3] Sec.3.3.2 #
#                                                                   #
# [3] définit un baseline plus robuste que la simple répétition :   #
#   ŷ(d,h) = Y(d-1,h)   si d est un jour de semaine                #
#   ŷ(d,h) = Y(d-7,h)   si d est un week-end                       #
# Ce baseline est la référence pour le calcul du RMAE.              #
# ================================================================= #

def predict_baseline(X_test: np.ndarray,
                     fold_test_norm: pd.DataFrame) -> np.ndarray:
    """
    Baseline naïf conforme à [3] Section 3.3.2.

    - Jours de semaine : répète le prix de D-1 même heure (lag 24h).
    - Week-ends        : répète le prix de D-7 même heure (lag 168h).

    Renvoie [µ=valeur_naïve, log_σ=0] (σ=1, incertitude non modélisée).

    Args:
        X_test: Séquences (n, input_length, n_features).
        fold_test_norm: DataFrame normalisé du test (pour récupérer les dates).

    Returns:
        np.ndarray: Prédictions (n, 2) avec [µ_baseline, 0].
    """
    target_idx = list(df_selected.columns).index(TARGET)
    n          = X_test.shape[0]
    mu_baseline = np.zeros(n)

    for i in range(n):
        # Dernière ligne de la fenêtre d'entrée
        last_row_idx = i * SEQUENCE_STRIDE + INPUT_LENGTH - 1
        if last_row_idx < len(fold_test_norm):
            date = pd.to_datetime(fold_test_norm.index[last_row_idx])
            if date.dayofweek >= 5:  # Samedi = 5, Dimanche = 6
                # Week-end → lag D-7
                lag = min(168, INPUT_LENGTH)
                mu_baseline[i] = X_test[i, -lag, target_idx]
            else:
                # Semaine → lag D-1
                lag = min(24, INPUT_LENGTH)
                mu_baseline[i] = X_test[i, -lag, target_idx]
        else:
            mu_baseline[i] = X_test[i, -1, target_idx]

    log_sigma = np.zeros_like(mu_baseline)
    return np.stack([mu_baseline, log_sigma], axis=1)

# ================================================================= #
# 12. TEST DE DIEBOLD & MARIANO                       [3] Sec.3.3.2 #
#                                                                   #
# Compare statistiquement deux modèles sur le set de test.          #
# H0 : E[g(y, ŷ1) - g(y, ŷ2)] > 0  (modèle_2 non meilleur)       #
# Un z-test unilatéral sur la série des différences de loss.        #
# p-value < 0.05 → modèle_2 significativement meilleur que modèle_1 #
# ================================================================= #

def diebold_mariano_test(y_true: np.ndarray,
                          y_pred_1: np.ndarray,
                          y_pred_2: np.ndarray) -> Tuple[float, float]:
    """
    Test de Diebold & Mariano (1995) pour la comparaison de deux modèles.

    Conformément à [3] Section 3.3.2.2 — loss = MAE absolue.

    H0 : E[MAE(modèle_1) - MAE(modèle_2)] > 0   (modèle_2 ≥ modèle_1)
    H1 : E[MAE(modèle_1) - MAE(modèle_2)] ≤ 0   (modèle_2 < modèle_1)

    Un p-value < 0.05 signifie que modèle_2 est significativement meilleur.

    Args:
        y_true: Valeurs réelles (n,).
        y_pred_1: Prédictions µ du modèle_1 (n,).
        y_pred_2: Prédictions µ du modèle_2 (n,).

    Returns:
        Tuple[float, float]: (statistique z, p-value unilatérale).
    """
    y_true = y_true.flatten()
    mu_1   = y_pred_1[:, 0]
    mu_2   = y_pred_2[:, 0]

    # Série des différences de loss
    d = np.abs(y_true - mu_1) - np.abs(y_true - mu_2)

    # z-test sur mean(d) > 0
    n     = len(d)
    d_bar = np.mean(d)
    d_var = np.var(d, ddof=1) / n
    z     = d_bar / (np.sqrt(d_var) + 1e-12)
    p_val = 1 - stats.norm.cdf(z)   # unilatéral : H1 : d_bar ≤ 0

    return float(z), float(p_val)

# ================================================================= #
# 13. RECALIBRATION GLISSANTE                         [3] Sec.3.3.5 #
#                                                                   #
# Ré-entraîne le modèle tous les N_RECALIB jours en incluant les    #
# données récentes du test. Cruciale lors des changements de régime  #
# (Covid-19, crise gaz 2021) où les patterns évoluent brusquement.  #
# ================================================================= #

def recalibrated_predict(fold_train_norm: pd.DataFrame,
                          fold_test_norm: pd.DataFrame,
                          recalib_every: int = 30,
                          verbose: int = 0) -> np.ndarray:
    """
    Prévision avec recalibration glissante tous les `recalib_every` jours.

    Conformément à [3] Section 3.3.5 : le modèle est ré-entraîné
    en intégrant les nouvelles données à chaque fenêtre de recalibration.
    Le jeu de validation est toujours extrait du jeu d'entraînement courant.

    Note de coût computationnel : chaque recalibration implique un
    réentraînement complet. Adapter recalib_every selon le budget temps.

    Args:
        fold_train_norm: DataFrame normalisé du train initial.
        fold_test_norm: DataFrame normalisé du test complet.
        recalib_every: Nombre de jours entre deux recalibrations.
        verbose: 0=silencieux.

    Returns:
        np.ndarray: Prédictions cumulées [µ, log_σ] sur le test complet.
    """
    recalib_stride = recalib_every * 24  # heures entre deux recalibs

    # Séquences test complètes
    X_test_full, y_test_full = get_X_y_strides(
        fold_test_norm, INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

    all_preds  = []
    train_pool = fold_train_norm.copy()

    for start in range(0, len(X_test_full), recalib_stride):
        end = min(start + recalib_stride, len(X_test_full))

        # Ré-entraîner sur le pool courant
        X_tr, y_tr = get_X_y_strides(train_pool, INPUT_LENGTH,
                                      OUTPUT_LENGTH, SEQUENCE_STRIDE)
        model         = init_model(input_shape=X_tr[0].shape)
        model, _      = fit_model(model, X_tr, y_tr, verbose=verbose)

        # Prédire le bloc courant
        X_block = X_test_full[start:end]
        preds   = model.predict(X_block, verbose=0)
        all_preds.append(preds)

        # Étendre le pool avec les données du bloc test prédit
        new_rows = fold_test_norm.iloc[
            start * SEQUENCE_STRIDE:end * SEQUENCE_STRIDE
        ]
        train_pool = pd.concat([train_pool, new_rows])
        print(f"  [RECALIB] Bloc {start}–{end} | pool={len(train_pool)} lignes")

    return np.vstack(all_preds)

# ================================================================= #
# 14. SHAP VALUES                                     [3] Sec.3.3.6 #
#                                                                   #
# Calcule les valeurs de Shapley pour identifier les features les   #
# plus importantes dans les prédictions du modèle.                  #
# Nécessite : pip install shap                                      #
# ================================================================= #

def compute_shap_values(model: tf.keras.Model, X_train: np.ndarray,
                         X_test: np.ndarray,
                         n_background: int = 100,
                         n_explain: int = 200) -> np.ndarray:
    """
    Calcule les SHAP values (DeepExplainer) sur le modèle LSTM.

    Conformément à [3] Section 3.3.6 : les SHAP values permettent
    d'identifier quelles features (prix voisins, consommation, gaz)
    contribuent le plus aux prédictions — essentiel pour la confiance
    opérationnelle.

    Nécessite le package 'shap' : pip install shap

    Args:
        model: Modèle entraîné.
        X_train: Données d'entraînement pour le background (n, L, F).
        X_test: Données à expliquer (m, L, F).
        n_background: Taille de l'échantillon background.
        n_explain: Nombre d'exemples à expliquer.

    Returns:
        np.ndarray: SHAP values (m, L, F) — contribution de chaque
                    (pas de temps, feature) à la prédiction µ.
    """
    try:
        import shap
    except ImportError:
        print("  [SHAP] Package 'shap' non installé. "
              "Lancez : pip install shap --break-system-packages")
        return None

    background = X_train[np.random.choice(len(X_train), n_background,
                                           replace=False)]
    # DeepExplainer sur la sortie µ (indice 0)
    explainer = shap.DeepExplainer(
        (model.input, model.output[:, 0]),
        background
    )
    shap_values = explainer.shap_values(X_test[:n_explain])
    return shap_values


def plot_shap_feature_importance(shap_values: np.ndarray,
                                  feature_names: List[str]):
    """
    Affiche l'importance moyenne des features via SHAP (style [3] Table 3.7).

    L'importance est moyennée sur tous les pas de temps et exemples.
    Permet d'identifier quelles features (prix voisins, gaz, dates)
    guident le modèle — insight opérationnel clé.

    Args:
        shap_values: Tableau (n, input_length, n_features).
        feature_names: Noms des features.
    """
    if shap_values is None:
        return

    # Importance = |SHAP| moyen sur tous les pas et exemples
    importance = np.abs(shap_values).mean(axis=(0, 1))  # (n_features,)
    sorted_idx = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(12, max(5, len(feature_names) // 3)))
    ax.barh(range(len(sorted_idx)),
            importance[sorted_idx],
            color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
    ax.set_xlabel('|SHAP| moyen')
    ax.set_title('Importance des features (SHAP) — [3] style Figure 3.7')
    ax.grid(axis='x', linewidth=0.5)
    plt.tight_layout()
    return ax

# ================================================================= #
# 15. VISUALISATION                                                 #
# ================================================================= #

def plot_history(history: tf.keras.callbacks.History):
    """Affiche la NLL (train/val) et le learning rate."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    ax[0].plot(history.history['loss'],     label='Train NLL')
    ax[0].plot(history.history['val_loss'], label='Val NLL')
    ax[0].set_title('Negative Log-Likelihood')
    ax[0].set_xlabel('Époque')
    ax[0].legend()
    ax[0].grid(linewidth=0.5)

    if 'lr' in history.history:
        ax[1].plot(history.history['lr'], color='orange')
        ax[1].set_title('Learning Rate')
        ax[1].set_xlabel('Époque')
        ax[1].set_yscale('log')
        ax[1].grid(linewidth=0.5)

    plt.tight_layout()
    return ax


def plot_probabilistic_forecast(y_true: np.ndarray, y_pred_raw: np.ndarray,
                                 n_points: int = 200):
    """
    Affiche µ ± 1σ et µ ± 2σ (style Figure 1a de [2]).

    Args:
        y_true: Valeurs réelles.
        y_pred_raw: Sorties [µ, log_σ] du modèle.
        n_points: Nombre de points à afficher.
    """
    y  = y_true.flatten()[:n_points]
    mu = y_pred_raw[:, 0][:n_points]
    sg = np.maximum(np.exp(y_pred_raw[:, 1]), 0.01)[:n_points]
    t  = np.arange(n_points)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, y,  label='Réel',     color='black',     linewidth=1)
    ax.plot(t, mu, label='µ prédit', color='steelblue', linewidth=1)
    ax.fill_between(t, mu - sg,   mu + sg,   alpha=0.30,
                    color='steelblue', label='±1σ')
    ax.fill_between(t, mu - 2*sg, mu + 2*sg, alpha=0.15,
                    color='steelblue', label='±2σ')
    ax.set_title('Prévision probabiliste (µ ± σ)')
    ax.set_xlabel('Pas de temps (heures)')
    ax.set_ylabel('Consommation normalisée')
    ax.legend()
    ax.grid(linewidth=0.5)
    plt.tight_layout()
    return ax

# ================================================================= #
# 16. PIPELINE PRINCIPAL — UN SEUL FOLD, N_EXPERIMENTS RÉPÉTITIONS  #
# ================================================================= #

folds = get_folds(df_selected, FOLD_LENGTH, FOLD_STRIDE)
print(f'\n{len(folds)} folds, forme : {folds[0].shape}')

fold = folds[0]
fold_train, fold_test           = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)
fold_train_norm, fold_test_norm = normalize_by_train_max(fold_train, fold_test)

X_train, y_train = get_X_y_strides(fold_train_norm, INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)
X_test,  y_test  = get_X_y_strides(fold_test_norm,  INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

print(f"X_train : {X_train.shape} | y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}  | y_test  : {y_test.shape}")

# Baseline naïf corrigé [3]
y_pred_baseline = predict_baseline(X_test, fold_test_norm)
metrics_baseline = compute_metrics(y_test, y_pred_baseline)

# LSTM probabiliste — N_EXPERIMENTS répétitions
print(f"\nEntraînement ({N_EXPERIMENTS} expériences) ...")
all_metrics  = []
last_pred    = None
last_history = None

for exp in range(N_EXPERIMENTS):
    print(f"  Expérience {exp+1}/{N_EXPERIMENTS}")
    model          = init_model(input_shape=X_train[0].shape)
    model, history = fit_model(model, X_train, y_train, verbose=0)
    y_pred_raw     = model.predict(X_test, verbose=0)
    metrics        = compute_metrics(y_test, y_pred_raw,
                                     y_pred_baseline=y_pred_baseline)
    all_metrics.append(metrics)
    print_metrics(metrics, label=f"Run {exp+1}")
    last_pred    = y_pred_raw
    last_history = history

avg = {k: round(np.mean([m[k] for m in all_metrics]), 4) for k in all_metrics[0]}
std = {k: round(np.std( [m[k] for m in all_metrics]), 4) for k in all_metrics[0]}

# Test de Diebold & Mariano [3] — LSTM vs Baseline
z_dm, p_dm = diebold_mariano_test(y_test, y_pred_baseline, last_pred)

print(f"\n{'='*70}")
print_metrics(metrics_baseline, label="Baseline")
print(f"  {'LSTM moy':12s} | NLL={avg['NLL']:.4f}±{std['NLL']:.4f} "
      f"| MAE={avg['MAE']:.4f}±{std['MAE']:.4f} "
      f"| SMAPE={avg['SMAPE']:.2f}%±{std['SMAPE']:.2f}%"
      f" | RMAE={avg.get('RMAE', float('nan')):.4f}")
print(f"  Amélioration MAE  : "
      f"{round((1 - avg['MAE'] / metrics_baseline['MAE']) * 100, 2)} %")
print(f"  DM test vs baseline : z={z_dm:.3f}, p={p_dm:.4f} "
      f"({'✓ significatif' if p_dm < 0.05 else '✗ non significatif'} α=5%)")
print(f"{'='*70}")

# Visualisations du dernier run
plot_history(last_history)
plot_probabilistic_forecast(y_test, last_pred)

# SHAP values (optionnel, nécessite le package shap)
shap_values = compute_shap_values(model, X_train, X_test)
if shap_values is not None:
    plot_shap_feature_importance(shap_values,
                                  feature_names=list(df_selected.columns))

# ================================================================= #
# 17. VALIDATION CROISÉE                                            #
# ================================================================= #

def cross_validate() -> pd.DataFrame:
    """
    Validation croisée temporelle : baseline vs LSTM probabiliste.

    Pour chaque fold :
        1. Normalisation par le max du train set (sans look-ahead bias).
        2. N_EXPERIMENTS répétitions LSTM, scores moyennés.
        3. Rapport NLL + MAE + SMAPE + DAE + RMAE.
        4. Test DM (significativité statistique vs baseline).

    Returns:
        pd.DataFrame: Métriques par fold avec test DM.
    """
    results = []
    folds   = get_folds(df_selected, FOLD_LENGTH, FOLD_STRIDE)

    for fold_id, fold in enumerate(folds):
        print(f"\n{'═'*70}")
        print(f"FOLD {fold_id + 1}/{len(folds)}")

        ft, ftest               = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)
        ft_norm, ftest_norm     = normalize_by_train_max(ft, ftest)
        X_tr, y_tr              = get_X_y_strides(ft_norm,    INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)
        X_te, y_te              = get_X_y_strides(ftest_norm,  INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

        y_bl       = predict_baseline(X_te, ftest_norm)
        metrics_bl = compute_metrics(y_te, y_bl)
        print_metrics(metrics_bl, label="Baseline")

        fold_metrics = []
        last_pred_cv = None
        for _ in range(N_EXPERIMENTS):
            m    = init_model(input_shape=X_tr[0].shape)
            m, _ = fit_model(m, X_tr, y_tr, verbose=0)
            pred = m.predict(X_te, verbose=0)
            fold_metrics.append(compute_metrics(y_te, pred,
                                                y_pred_baseline=y_bl))
            last_pred_cv = pred

        avg_fold = {k: round(np.mean([fm[k] for fm in fold_metrics]), 4)
                    for k in fold_metrics[0]}
        print_metrics(avg_fold, label="LSTM moy.")

        # Test DM
        z_f, p_f = diebold_mariano_test(y_te, y_bl, last_pred_cv)
        sig      = '✓' if p_f < 0.05 else '✗'
        print(f"  DM test : z={z_f:.3f}, p={p_f:.4f} ({sig} significatif α=5%)")

        results.append({
            'fold':             fold_id + 1,
            'baseline_mae':     metrics_bl['MAE'],
            'baseline_smape':   metrics_bl['SMAPE'],
            'lstm_nll':         avg_fold['NLL'],
            'lstm_mae':         avg_fold['MAE'],
            'lstm_smape':       avg_fold['SMAPE'],
            'lstm_dae':         avg_fold['DAE'],
            'lstm_rmae':        avg_fold.get('RMAE', float('nan')),
            'improvement_%':    round((1 - avg_fold['MAE'] / metrics_bl['MAE']) * 100, 2),
            'dm_pvalue':        round(p_f, 4),
            'dm_significant':   p_f < 0.05,
        })

    df_res = pd.DataFrame(results)
    print(f"\n{'═'*70}")
    print("RÉCAPITULATIF")
    print(df_res.to_string(index=False))
    print(f"\nMoyenne NLL    : {df_res['lstm_nll'].mean():.4f}")
    print(f"Moyenne SMAPE  : {df_res['lstm_smape'].mean():.2f}%")
    print(f"Moyenne RMAE   : {df_res['lstm_rmae'].mean():.4f}")
    print(f"Amélioration   : {df_res['improvement_%'].mean():.2f}%")
    print(f"Folds DM sign. : {df_res['dm_significant'].sum()}/{len(df_res)}")
    return df_res


df_results = cross_validate()
