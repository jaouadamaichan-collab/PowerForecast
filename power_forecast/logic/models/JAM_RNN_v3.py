"""
Power Forecast - Modèle LSTM Optimisé (v3)
==========================================
Sources scientifiques :
  [1] Li & Becker, NTNU (2021) — "Day-ahead electricity price prediction applying
      hybrid models of LSTM-based deep learning methods and feature selection
      algorithms under consideration of market coupling"
  [2] Trebbien et al., Forschungszentrum Jülich (2023) — "Probabilistic Forecasting
      of Day-Ahead Electricity Prices and their Volatility with LSTMs"

─────────────────────────────────────────────────────────────────────────────
APPORTS DE [1] — Li & Becker (2021)
─────────────────────────────────────────────────────────────────────────────
    [FEATURE SELECTION]
    1. Sélection de features par régression Lasso (méthode M5 du papier).
       Trop de features redondantes dégradent le LSTM (malédiction de la
       dimensionnalité). Lasso élimine les features peu informatives en
       réduisant leur coefficient à zéro. M5 (Lasso-LSTM) obtient
       SMAPE=4.89, meilleur score individuel (Table 5).

    [ARCHITECTURE]
    2. Architecture LSTM-LSTM Encoder-Decoder (M6 du papier).
       LSTM-LSTM bat CNN-LSTM et ConvLSTM pour les séries temporelles.

    [ÉVALUATION]
    3. N_EXPERIMENTS = 10 répétitions puis moyenne (section 4.2.4).
       L'initialisation aléatoire fait varier les résultats de ±1% SMAPE.

─────────────────────────────────────────────────────────────────────────────
APPORTS DE [2] — Trebbien et al. (2023)
─────────────────────────────────────────────────────────────────────────────
    [PRÉVISION PROBABILISTE] — Apport principal du papier
    4. Le modèle prédit µ ET σ (moyenne + écart-type) simultanément.
       Cela quantifie l'incertitude de chaque prédiction — crucial pour
       la gestion de portefeuille, le demand-side management et le stockage.
       La sortie σ donne un intervalle de confiance à chaque pas de temps.

    [LOSS NLL]
    5. La loss MSE est remplacée par la Negative Log-Likelihood (NLL) :
           NLL = mean[ log(2π σ²)/2 + (y - µ)²/(2σ²) ]
       NLL entraîne conjointement µ et σ : le modèle est pénalisé s'il
       prédit un σ trop large (imprécis) ou trop étroit (sous-estime l'incert.).
       On prédit log(σ) pour garantir σ > 0, avec σ_min = 0.01.

    [INPUT_LENGTH = 96h]
    6. Fenêtre réduite à 96h (4 jours) au lieu de 14 jours.
       Justification physique (section IV du papier) : à l'échelle de 4 jours,
       les prix électriques suivent une distribution gaussienne — hypothèse
       nécessaire à la validité de la loss NLL. Au-delà, la volatilité
       introduit des queues lourdes incompatibles avec une loi gaussienne.

    [ARCHITECTURE LÉGÈRE]
    7. Architecture simplifiée : depth=2, width=32, dropout=0.2.
       Le papier montre qu'un modèle léger avec early stopping agressif
       (patience=200) suffit pour atteindre des performances SOTA.

    [NORMALISATION SANS LOOK-AHEAD BIAS]
    8. Normalisation par le maximum du train set uniquement (section III).
       La couche Normalization de Keras est retirée : elle calcule les stats
       sur l'ensemble des données et introduirait un look-ahead bias.
       Chaque fold est normalisé indépendamment par son propre train set.

    [MÉTRIQUES]
    9. Évaluation par NLL (probabiliste) + MAE + SMAPE (point forecast).
       Le NLL mesure la qualité de l'intervalle de confiance prédit.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

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
# 2. FEATURE SELECTION PAR LASSO                          [1]       #
#                                                                   #
# Lasso minimise : MSE + alpha * sum(|coef|)                        #
# Les features peu informatives ont leur coef réduit à 0.           #
# alpha=0.01 proche du lambda=0.02 utilisé dans [1].                #
# ================================================================= #

def lasso_feature_selection(df: pd.DataFrame, target: str,
                             alpha: float = 0.01) -> List[str]:
    """
    Sélectionne les features pertinentes pour prédire `target` via Lasso.

    Args:
        df (pd.DataFrame): DataFrame avec toutes les features.
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


selected_features = lasso_feature_selection(df, target='FRA', alpha=0.01)
df_selected       = df[selected_features + ['FRA']]
print(f"Shape après sélection : {df_selected.shape}")

# ================================================================= #
# 3. CONFIGURATION GLOBALE                                          #
# ================================================================= #

TARGET          = 'FRA'
N_FEATURES      = df_selected.shape[1]

FOLD_LENGTH      = 24 * 365 * 2   # 2 ans (~17 520h, proche des 17 000h de [2])
FOLD_STRIDE      = 24 * 91        # 1 trimestre entre chaque fold
TRAIN_TEST_RATIO = 0.7

# INPUT_LENGTH = 96h — fenêtre physiquement motivée par [2] (section IV) :
# à 4 jours, les prix sont gaussiens → hypothèse NLL valide.
INPUT_LENGTH    = 96    # 4 jours
OUTPUT_LENGTH   = 1
SEQUENCE_STRIDE = 24    # 1 pas/jour
DAY_AHEAD_GAP   = 24    # Prévision J+1

N_EXPERIMENTS   = 10    # Répétitions pour stabiliser les scores [1]

print(f"N_FEATURES = {N_FEATURES} | INPUT_LENGTH = {INPUT_LENGTH}h = {INPUT_LENGTH//24} jours")

# ================================================================= #
# 4. FONCTIONS DE DÉCOUPAGE                                         #
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
    Normalise train et test par le maximum absolu de chaque feature dans le train.

    Conformément à [2] (section III) : normalisation exclusivement sur le
    train set pour éviter tout look-ahead bias. Les features constantes
    (max=0) sont laissées à 0.

    Args:
        fold_train (pd.DataFrame): Données d'entraînement.
        fold_test (pd.DataFrame): Données de test.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train normalisé, test normalisé)
    """
    max_vals = fold_train.abs().max()
    max_vals = max_vals.replace(0, 1)   # Éviter la division par zéro
    return fold_train / max_vals, fold_test / max_vals


def get_X_y_strides(fold: pd.DataFrame, input_length: int, output_length: int,
                    sequence_stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère (X_i, y_i) par fenêtre glissante avec décalage day-ahead.

    Schéma : [--- X_i ---][--- gap 24h ---][y_i]

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
# 5. LOSS NLL (NEGATIVE LOG-LIKELIHOOD)               [2]           #
#                                                                   #
# Le modèle prédit [µ, log_σ]. La NLL entraîne µ et σ              #
# conjointement (équation 2 du papier) :                            #
#   NLL = mean[ log(2π σ²)/2 + (y − µ)²/(2σ²) ]                   #
#                                                                   #
# On prédit log_σ (et non σ directement) pour que σ soit           #
# toujours positif sans contrainte hard dans le réseau.             #
# ================================================================= #

def nll_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Negative Log-Likelihood gaussienne.

    y_pred[:, 0] = µ     (prédiction ponctuelle)
    y_pred[:, 1] = log_σ (log de l'écart-type → σ = exp(log_σ) ≥ 0.01)

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
# 6. ARCHITECTURE LSTM PROBABILISTE                   [1] + [2]     #
#                                                                   #
# Combine :                                                         #
#   - Encoder-Decoder LSTM-LSTM de [1] (M6, section 3.4.1)         #
#   - Sortie probabiliste µ + log_σ de [2] (section III)           #
#   - Hyperparamètres légers de [2] : width=32, dropout=0.2        #
#   - Pas de couche Normalization Keras (look-ahead bias supprimé)  #
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
# 7. ENTRAÎNEMENT                                                   #
# ================================================================= #

def fit_model(model: tf.keras.Model, X_train: np.ndarray,
              y_train: np.ndarray, verbose: int = 1):
    """
    Entraîne le modèle avec early stopping (patience=200) et ReduceLROnPlateau.

    La patience élevée (200) est conforme à [2] section V : le modèle
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
# 8. MÉTRIQUES                                        [1] + [2]     #
# ================================================================= #

def compute_metrics(y_true: np.ndarray,
                    y_pred_raw: np.ndarray) -> Dict[str, float]:
    """
    Calcule NLL, MAE et SMAPE.

    y_pred_raw[:, 0] = µ      → utilisé pour MAE et SMAPE
    y_pred_raw[:, 1] = log_σ  → µ + σ utilisés pour NLL

    Args:
        y_true: Valeurs réelles (n,).
        y_pred_raw: Sorties brutes [µ, log_σ] (n, 2).

    Returns:
        Dict[str, float]: NLL, MAE, SMAPE.
    """
    y_true = y_true.flatten()
    mu     = y_pred_raw[:, 0]
    sigma  = np.maximum(np.exp(y_pred_raw[:, 1]), 0.01)

    nll   = float(np.mean(np.log(2 * np.pi * sigma**2) / 2
                          + (y_true - mu)**2 / (2 * sigma**2)))
    mae   = float(np.mean(np.abs(y_true - mu)))
    smape = float(100 * np.mean(np.abs(y_true - mu) /
                                ((np.abs(y_true) + np.abs(mu)) / 2 + 1e-8)))

    return {'NLL': round(nll, 4), 'MAE': round(mae, 4), 'SMAPE': round(smape, 4)}


def print_metrics(metrics: Dict[str, float], label: str = ""):
    """Affiche les métriques de manière formatée."""
    print(f"  {label:12s} | NLL={metrics['NLL']:7.4f} "
          f"| MAE={metrics['MAE']:8.4f} | SMAPE={metrics['SMAPE']:.2f}%")

# ================================================================= #
# 9. BASELINE NAÏF                                                  #
# ================================================================= #

def predict_baseline(X_test: np.ndarray) -> np.ndarray:
    """
    Baseline naïf : dernière valeur de TARGET dans la fenêtre X.

    Renvoie [µ=dernière_valeur, log_σ=0] pour chaque séquence,
    soit σ=1 (incertitude non modélisée).

    Args:
        X_test: Séquences (n, input_length, n_features).

    Returns:
        np.ndarray: Prédictions (n, 2) avec [µ_baseline, log(1)=0].
    """
    target_idx  = list(df_selected.columns).index(TARGET)
    mu_baseline = X_test[:, -1, target_idx]
    log_sigma   = np.zeros_like(mu_baseline)   # σ=1 → log_σ=0
    return np.stack([mu_baseline, log_sigma], axis=1)

# ================================================================= #
# 10. VISUALISATION                                                 #
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
# 11. PIPELINE PRINCIPAL — UN SEUL FOLD, N_EXPERIMENTS RÉPÉTITIONS  #
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

# Baseline
metrics_baseline = compute_metrics(y_test, predict_baseline(X_test))

# LSTM probabiliste — N_EXPERIMENTS répétitions
print(f"\nEntraînement ({N_EXPERIMENTS} expériences) ...")
all_metrics = []

for exp in range(N_EXPERIMENTS):
    print(f"  Expérience {exp+1}/{N_EXPERIMENTS}")
    model          = init_model(input_shape=X_train[0].shape)
    model, history = fit_model(model, X_train, y_train, verbose=0)
    y_pred_raw     = model.predict(X_test, verbose=0)
    metrics        = compute_metrics(y_test, y_pred_raw)
    all_metrics.append(metrics)
    print_metrics(metrics, label=f"Run {exp+1}")

avg = {k: round(np.mean([m[k] for m in all_metrics]), 4) for k in all_metrics[0]}
std = {k: round(np.std( [m[k] for m in all_metrics]), 4) for k in all_metrics[0]}

print(f"\n{'='*65}")
print_metrics(metrics_baseline, label="Baseline")
print(f"  {'LSTM moy':12s} | NLL={avg['NLL']:.4f}±{std['NLL']:.4f} "
      f"| MAE={avg['MAE']:.4f}±{std['MAE']:.4f} "
      f"| SMAPE={avg['SMAPE']:.2f}%±{std['SMAPE']:.2f}%")
print(f"  🔥 Amélioration MAE : "
      f"{round((1 - avg['MAE'] / metrics_baseline['MAE']) * 100, 2)} %")
print(f"{'='*65}")

# Visualisations du dernier run
plot_history(history)
plot_probabilistic_forecast(y_test, y_pred_raw)

# ================================================================= #
# 12. VALIDATION CROISÉE                                            #
# ================================================================= #

def cross_validate() -> pd.DataFrame:
    """
    Validation croisée temporelle : baseline vs LSTM probabiliste.

    Pour chaque fold :
        1. Normalisation par le max du train set (sans look-ahead bias).
        2. N_EXPERIMENTS répétitions LSTM, scores moyennés.
        3. Rapport NLL + MAE + SMAPE.

    Returns:
        pd.DataFrame: Métriques par fold.
    """
    results = []
    folds   = get_folds(df_selected, FOLD_LENGTH, FOLD_STRIDE)

    for fold_id, fold in enumerate(folds):
        print(f"\n{'═'*65}")
        print(f"FOLD {fold_id + 1}/{len(folds)}")

        ft, ftest               = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)
        ft_norm, ftest_norm     = normalize_by_train_max(ft, ftest)
        X_tr, y_tr              = get_X_y_strides(ft_norm,    INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)
        X_te, y_te              = get_X_y_strides(ftest_norm,  INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

        metrics_bl = compute_metrics(y_te, predict_baseline(X_te))
        print_metrics(metrics_bl, label="Baseline")

        fold_metrics = []
        for _ in range(N_EXPERIMENTS):
            m    = init_model(input_shape=X_tr[0].shape)
            m, _ = fit_model(m, X_tr, y_tr, verbose=0)
            fold_metrics.append(compute_metrics(y_te, m.predict(X_te, verbose=0)))

        avg_fold = {k: round(np.mean([fm[k] for fm in fold_metrics]), 4)
                    for k in fold_metrics[0]}
        print_metrics(avg_fold, label="LSTM moy.")
        print(f"  🏋🏽 Amélioration MAE : "
              f"{round((1 - avg_fold['MAE'] / metrics_bl['MAE']) * 100, 2)} %")

        results.append({
            'fold':           fold_id + 1,
            'baseline_mae':   metrics_bl['MAE'],
            'baseline_smape': metrics_bl['SMAPE'],
            'lstm_nll':       avg_fold['NLL'],
            'lstm_mae':       avg_fold['MAE'],
            'lstm_smape':     avg_fold['SMAPE'],
            'improvement_%':  round((1 - avg_fold['MAE'] / metrics_bl['MAE']) * 100, 2)
        })

    df_res = pd.DataFrame(results)
    print(f"\n{'═'*65}")
    print("RÉCAPITULATIF")
    print(df_res.to_string(index=False))
    print(f"\nMoyenne NLL   : {df_res['lstm_nll'].mean():.4f}")
    print(f"Moyenne SMAPE : {df_res['lstm_smape'].mean():.2f}%")
    print(f"Amélioration  : {df_res['improvement_%'].mean():.2f}%")
    return df_res


df_results = cross_validate()
