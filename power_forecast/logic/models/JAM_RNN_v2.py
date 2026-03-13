"""
Power Forecast - Modèle LSTM Optimisé (v2 — inspiré de Li & Becker, 2021)
==========================================================================
Améliorations issues de l'étude scientifique :
"Day-ahead electricity price prediction applying hybrid models of LSTM-based
deep learning methods and feature selection algorithms under consideration
of market coupling" (Li & Becker, NTNU, 2021)

    [FEATURE SELECTION] — Apport principal du papier
    1. Sélection de features par régression Lasso (méthode M5 du papier).
       Le papier montre que trop de features redondantes DÉGRADENT le LSTM
       (malédiction de la dimensionnalité). Lasso élimine automatiquement les
       features peu informatives en réduisant leur coefficient à zéro.
       → Dans le papier, M5 (Lasso-LSTM) obtient SMAPE=4.89, le meilleur score
         parmi tous les modèles testés (hors two-stage).

    [ARCHITECTURE TWO-STAGE] — Meilleure architecture du papier
    2. Architecture LSTM-LSTM Encoder-Decoder (M6 du papier) :
       un LSTM encodeur compresse la séquence en vecteur latent,
       un LSTM décodeur produit la prédiction depuis ce vecteur.
       Le papier montre que LSTM-LSTM bat CNN-LSTM et ConvLSTM pour les
       séries temporelles (les CNN sont conçus pour les images, pas les séries).

    [ÉVALUATION ROBUSTE] — Pratique du papier
    3. Métriques multiples : MAE, RMSE, MAPE, SMAPE (comme dans le papier).
       La MAE seule est insuffisante — le SMAPE est particulièrement utile
       car il est symétrique et ne diverge pas quand les valeurs cibles → 0.
    4. N_EXPERIMENTS = 10 répétitions puis moyenne des scores.
       Le papier montre que l'initialisation aléatoire des poids LSTM peut
       faire varier les résultats de ±1% SMAPE. La moyenne de 10 runs donne
       une estimation fiable et reproductible.

    [FENÊTRE D'ENTRÉE]
    5. INPUT_LENGTH = 14 jours, confirmé comme standard EPF par le papier
       (section 4.3.2 : "The input sequence length is 14 days (2 weeks,
       commonly used in EPF)"). On revient donc à 14 jours.

    [COHÉRENCE MÉTIER]
    7. Décalage DAY_AHEAD_GAP = 24h entre X_i et y_i (prévision J+1).
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from power_forecast.logic.get_data.download_api import build_feature_dataframe
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers, Input, optimizers
from tensorflow.keras.layers import Normalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

pd.set_option('display.max_columns', None)

# ================================================================= #
# 1. CHARGEMENT DES DONNÉES                                         #
# ================================================================= #

df = build_feature_dataframe('raw_data/all_countries.csv', load_from_pickle=False)

# ================================================================= #
# 2. FEATURE SELECTION PAR LASSO                                    #
#                                                                   #
# Inspiré de M5 (Lasso-LSTM) du papier — meilleure méthode         #
# individuelle avec SMAPE=4.89 (Table 5).                           #
#                                                                   #
# Principe : Lasso ajoute une pénalité L1 à la loss de régression. #
# Les features peu informatives voient leur coefficient réduit à 0  #
# et sont automatiquement éliminées. On conserve uniquement les     #
# features dont le coefficient Lasso est non nul.                   #
#                                                                   #
# Le papier sélectionne 30 features parmi 62. On adapte ce ratio   #
# à notre dataset.                                                  #
# ================================================================= #

def lasso_feature_selection(df: pd.DataFrame, target: str,
                             alpha: float = 0.01) -> List[str]:
    """
    Sélectionne les features les plus pertinentes pour prédire `target` via Lasso.

    La régression Lasso minimise :
        MSE + alpha * sum(|coefficients|)
    Les features avec coefficient = 0 sont éliminées.

    Un alpha faible → peu d'élimination (modèle permissif).
    Un alpha élevé → élimination agressive (modèle parcimonieux).
    La valeur alpha=0.01 est proche de celle utilisée dans le papier (lambda=0.02).

    Args:
        df (pd.DataFrame): DataFrame complet avec toutes les features.
        target (str): Nom de la colonne cible.
        alpha (float): Paramètre de régularisation Lasso (défaut=0.01).

    Returns:
        List[str]: Liste des noms de features sélectionnées (coefficients non nuls).
    """
    features = [c for c in df.columns if c != target]
    X = df[features].values
    y = df[target].values

    # Standardisation nécessaire pour que Lasso compare les features équitablement
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_scaled, y)

    # Récupération des features dont le coefficient est non nul
    selected = [features[i] for i, coef in enumerate(lasso.coef_) if coef != 0]
    print(f"\nLasso feature selection (alpha={alpha}) :")
    print(f"  {len(features)} features initiales → {len(selected)} features sélectionnées")
    print(f"  Features conservées : {selected}")
    return selected


selected_features = lasso_feature_selection(df, target='FRA', alpha=0.01)

# On conserve uniquement les features sélectionnées + la cible
df_selected = df[selected_features + ['FRA']]
print(f"\nShape après feature selection : {df_selected.shape}")

# ================================================================= #
# 3. CONFIGURATION GLOBALE                                          #
# ================================================================= #

TARGET          = 'FRA'
N_FEATURES      = df_selected.shape[1]

FOLD_LENGTH      = 24 * 365 * 2   # 2 ans de données horaires
FOLD_STRIDE      = 24 * 91        # 1 trimestre entre chaque fold
TRAIN_TEST_RATIO = 0.7

# INPUT_LENGTH = 14 jours — standard EPF confirmé par le papier (section 4.3.2)
INPUT_LENGTH    = 24 * 14   # 336 pas de temps
OUTPUT_LENGTH   = 1
SEQUENCE_STRIDE = 24        # 1 jour entre chaque séquence
DAY_AHEAD_GAP   = 24        # Décalage day-ahead : on prédit t+24h

# Nombre de répétitions pour stabiliser les scores (pratique du papier section 4.2.4)
N_EXPERIMENTS   = 10

print(f"N_FEATURES après sélection : {N_FEATURES}")
print(f"INPUT_LENGTH = {INPUT_LENGTH} pas = {INPUT_LENGTH//24} jours")

# ================================================================= #
# 4. FONCTIONS DE DÉCOUPAGE                                         #
# ================================================================= #

def get_folds(df: pd.DataFrame, fold_length: int, fold_stride: int) -> List[pd.DataFrame]:
    """
    Parcourt le DataFrame pour extraire des folds de longueur fixe.

    Args:
        df (pd.DataFrame): Série temporelle complète.
        fold_length (int): Nombre de pas de temps par fold.
        fold_stride (int): Pas entre deux folds consécutifs.

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
                     input_length: int) -> Tuple[pd.DataFrame]:
    """
    Divise un fold en train et test avec chevauchement intentionnel.

    Args:
        fold (pd.DataFrame): Fold à diviser.
        train_test_ratio (float): Proportion allouée au train.
        input_length (int): Longueur d'une séquence X_i.

    Returns:
        Tuple[pd.DataFrame]: (fold_train, fold_test)
    """
    last_train_idx = round(train_test_ratio * len(fold))
    fold_train     = fold.iloc[0:last_train_idx, :]
    fold_test      = fold.iloc[last_train_idx - input_length:, :]
    return (fold_train, fold_test)


def get_X_y_strides(fold: pd.DataFrame, input_length: int, output_length: int,
                    sequence_stride: int) -> Tuple[np.array]:
    """
    Génère des paires (X_i, y_i) par fenêtre glissante avec décalage day-ahead.

    Schéma temporel : [--- X_i ---][--- gap 24h ---][y_i]

    Args:
        fold (pd.DataFrame): Fold de la série temporelle.
        input_length (int): Longueur de X_i.
        output_length (int): Longueur de y_i.
        sequence_stride (int): Pas entre séquences consécutives.

    Returns:
        Tuple[np.array]: (X, y)
    """
    X, y = [], []
    for i in range(0, len(fold), sequence_stride):
        if (i + input_length + DAY_AHEAD_GAP + output_length) > len(fold):
            break
        X.append(fold.iloc[i:i + input_length, :])
        y_start = i + input_length + DAY_AHEAD_GAP
        y.append(fold.iloc[y_start:y_start + output_length, :][[TARGET]])
    return (np.array(X), np.array(y))


# ================================================================= #
# 5. ARCHITECTURE LSTM-LSTM ENCODER-DECODER                         #
#                                                                   #
# Inspirée de M6 (LSTM-LSTM Encoder-Decoder) du papier.            #
# Le papier montre que M6 surpasse CNN-LSTM (M7) et ConvLSTM (M8)  #
# car les LSTM sont conçus pour les séries temporelles.             #
#                                                                   #
# Principe :                                                        #
# - L'encodeur LSTM lit la séquence X_i et la compresse en un      #
#   vecteur latent (état caché final).                              #
# - Le décodeur LSTM prend ce vecteur et produit la prédiction.     #
# ================================================================= #

def init_model(X_train: np.array, y_train: np.array) -> tf.keras.Model:
    """
    Construit l'architecture LSTM-LSTM Encoder-Decoder.

    Architecture (inspirée de M6 du papier, section 3.4.1) :
        Encodeur :
            - Normalisation des features
            - LSTM(300) avec return_sequences=False → produit le vecteur latent
              (300 unités comme dans le papier, section 4.3.2)
            - Dropout(0.2) pour régularisation

        Pont encodeur → décodeur :
            - RepeatVector(1) : répète le vecteur latent pour alimenter le décodeur

        Décodeur :
            - LSTM(100) avec return_sequences=False → interprète le vecteur latent
              (100 unités, comme la Dense layer du papier)
            - Dropout(0.2)
            - Dense(output_length, linear) → prédiction finale

    Compilation :
        - Loss : MSE (comme dans le papier)
        - Optimiseur : Adam (comme dans le papier)
        - Métrique : MAE

    Args:
        X_train (np.array): Entrées (n_samples, input_length, n_features).
        y_train (np.array): Cibles (n_samples, output_length, 1).

    Returns:
        tf.keras.Model: Modèle compilé.
    """
    normalizer = Normalization()
    normalizer.adapt(X_train)

    output_length = y_train.shape[1]

    model = models.Sequential([
        Input(shape=X_train[0].shape),
        normalizer,

        # === ENCODEUR ===
        # Lit toute la séquence et produit un vecteur latent (état caché final)
        layers.LSTM(300, activation='tanh', return_sequences=False),
        layers.Dropout(0.2),

        # Répète le vecteur latent pour le passer au décodeur LSTM
        # (le décodeur attend une séquence de forme (timesteps, features))
        layers.RepeatVector(1),

        # === DÉCODEUR ===
        # Interprète le vecteur latent et génère la représentation pour la prédiction
        layers.LSTM(100, activation='tanh', return_sequences=False),
        layers.Dropout(0.2),

        # Couche de sortie linéaire pour la régression
        layers.Dense(output_length, activation='linear'),
    ])

    model.compile(
        loss='mse',
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=['mae']
    )
    return model


# ================================================================= #
# 6. ENTRAÎNEMENT                                                   #
# ================================================================= #

def fit_model(model: tf.keras.Model, X_train: np.array, y_train: np.array,
              verbose: int = 1) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Entraîne le modèle avec early stopping et réduction adaptative du learning rate.

    Args:
        model (tf.keras.Model): Modèle Keras compilé.
        X_train (np.array): Données d'entraînement.
        y_train (np.array): Cibles d'entraînement.
        verbose (int): 0=silencieux, 1=barre de progression.

    Returns:
        Tuple: (modèle entraîné, historique)
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, mode='min',
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-6, verbose=verbose),
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=0.3,
        shuffle=False,
        batch_size=16,
        epochs=100,
        callbacks=callbacks,
        verbose=verbose
    )
    return model, history


# ================================================================= #
# 7. MÉTRIQUES MULTIPLES                                            #
#                                                                   #
# Le papier utilise MAE, RMSE, MAPE et SMAPE (section 4.1.1).      #
# La MAE seule est insuffisante — le SMAPE est particulièrement     #
# utile : il est symétrique et normalisé entre sur-/sous-estimation.#
# ================================================================= #

def compute_metrics(y_true: np.array, y_pred: np.array) -> Dict[str, float]:
    """
    Calcule les quatre métriques d'évaluation utilisées dans le papier.

    MAE   = mean(|y - ŷ|)
    RMSE  = sqrt(mean((y - ŷ)²))
    MAPE  = 100 * mean(|y - ŷ| / |y|)
    SMAPE = 100 * mean(|y - ŷ| / ((|y| + |ŷ|) / 2))   ← symétrique, robuste

    Args:
        y_true (np.array): Valeurs réelles.
        y_pred (np.array): Valeurs prédites.

    Returns:
        Dict[str, float]: Dictionnaire avec les quatre métriques.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mae   = np.mean(np.abs(y_true - y_pred))
    rmse  = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape  = 100 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)))
    smape = 100 * np.mean(np.abs(y_true - y_pred) /
                          ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8))
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
            'MAPE': round(mape, 4), 'SMAPE': round(smape, 4)}


def print_metrics(metrics: Dict[str, float], label: str = ""):
    """Affiche les métriques de manière formatée."""
    print(f"  {label:10s} | MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} "
          f"| MAPE={metrics['MAPE']:.2f}% | SMAPE={metrics['SMAPE']:.2f}%")


# ================================================================= #
# 8. BASELINE NAÏF                                                  #
# ================================================================= #

def init_baseline() -> tf.keras.Model:
    """
    Baseline naïf "dernière valeur observée".

    Renvoie directement la dernière valeur de TARGET dans la fenêtre d'entrée.
    Sert de borne inférieure : le LSTM doit impérativement faire mieux.

    Returns:
        tf.keras.Model: Modèle compilé.
    """
    # Index de TARGET dans df_selected
    target_idx = list(df_selected.columns).index(TARGET)
    model = models.Sequential([
        layers.Lambda(lambda x: x[:, -1, target_idx, None])
    ])
    model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.02), metrics=['mae'])
    return model


# ================================================================= #
# 9. VISUALISATION                                                 #
# ================================================================= #

def plot_history(history: tf.keras.callbacks.History):
    """
    Affiche MSE, MAE et learning rate au fil des époques.

    Args:
        history: Objet History retourné par model.fit().
    """
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    for a, key, title in zip(ax,
                              ['loss', 'mae', 'lr'],
                              ['Perte MSE', 'Métrique MAE', 'Learning Rate']):
        if key in history.history:
            a.plot(history.history[key], label='Train')
            if f'val_{key}' in history.history:
                a.plot(history.history[f'val_{key}'], label='Validation')
            a.set_title(title)
            a.set_xlabel('Époque')
            if key == 'lr':
                a.set_yscale('log')
            a.legend()
            a.grid(linewidth=0.5)
    plt.tight_layout()
    return ax


# ================================================================= #
# 10. PIPELINE PRINCIPAL — UN SEUL FOLD AVEC 10 RÉPÉTITIONS         #
#                                                                   #
# Inspiré de la section 4.2.4 du papier : "we employed ten         #
# experiments for each model to reduce the impact of variability".  #
# Les scores sont moyennés sur les 10 runs.                        #
# ================================================================= #

folds = get_folds(df_selected, FOLD_LENGTH, FOLD_STRIDE)
print(f'\n{len(folds)} folds générés, forme : {folds[0].shape}')

fold = folds[0]
(fold_train, fold_test) = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)

X_train, y_train = get_X_y_strides(fold_train, INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)
X_test,  y_test  = get_X_y_strides(fold_test,  INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

print(f"X_train : {X_train.shape} | y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}  | y_test  : {y_test.shape}")

# --- Baseline (pas besoin de répétition — déterministe) ---
baseline_model   = init_baseline()
y_pred_baseline  = baseline_model.predict(X_test, verbose=0)
metrics_baseline = compute_metrics(y_test, y_pred_baseline)
print(f"\n{'='*65}")
print_metrics(metrics_baseline, label="Baseline")

# --- LSTM Encoder-Decoder : N_EXPERIMENTS répétitions ---
print(f"\nEntraînement sur {N_EXPERIMENTS} expériences (initialisation aléatoire différente) ...")
all_metrics = []

for exp in range(N_EXPERIMENTS):
    print(f"\n  Expérience {exp+1}/{N_EXPERIMENTS}")
    model = init_model(X_train, y_train)
    model, history = fit_model(model, X_train, y_train, verbose=0)
    y_pred   = model.predict(X_test, verbose=0)
    metrics  = compute_metrics(y_test, y_pred)
    all_metrics.append(metrics)
    print_metrics(metrics, label=f"  Run {exp+1}")

# Moyenne et écart-type sur les N_EXPERIMENTS runs
avg_metrics = {k: round(np.mean([m[k] for m in all_metrics]), 4) for k in all_metrics[0]}
std_metrics = {k: round(np.std( [m[k] for m in all_metrics]), 4) for k in all_metrics[0]}

print(f"\n{'='*65}")
print_metrics(metrics_baseline, label="Baseline")
print(f"  {'LSTM moy':10s} | MAE={avg_metrics['MAE']:.4f}±{std_metrics['MAE']:.4f} "
      f"| RMSE={avg_metrics['RMSE']:.4f}±{std_metrics['RMSE']:.4f} "
      f"| MAPE={avg_metrics['MAPE']:.2f}%±{std_metrics['MAPE']:.2f}% "
      f"| SMAPE={avg_metrics['SMAPE']:.2f}%±{std_metrics['SMAPE']:.2f}%")
print(f"  🔥 Amélioration MAE : "
      f"{round((1 - avg_metrics['MAE'] / metrics_baseline['MAE']) * 100, 2)} %")
print(f"{'='*65}")

# Affichage de l'historique du dernier run (pour visualisation)
plot_history(history)


# ================================================================= #
# 11. VALIDATION CROISÉE AVEC MÉTRIQUES MULTIPLES                   #
# ================================================================= #

def cross_validate() -> pd.DataFrame:
    """
    Validation croisée temporelle comparant baseline et LSTM Encoder-Decoder.

    Pour chaque fold, entraîne N_EXPERIMENTS modèles LSTM et moyenne les scores,
    conformément à la pratique du papier (section 4.2.4).

    Returns:
        pd.DataFrame: Tableau récapitulatif des métriques par fold.
    """
    results = []
    folds   = get_folds(df_selected, FOLD_LENGTH, FOLD_STRIDE)

    for fold_id, fold in enumerate(folds):
        print(f"\n{'═'*65}")
        print(f"FOLD {fold_id + 1}/{len(folds)}")

        (fold_train, fold_test) = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)
        X_tr, y_tr = get_X_y_strides(fold_train, INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)
        X_te, y_te = get_X_y_strides(fold_test,  INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

        # Baseline
        bm            = init_baseline()
        y_pred_bl     = bm.predict(X_te, verbose=0)
        metrics_bl    = compute_metrics(y_te, y_pred_bl)
        print_metrics(metrics_bl, label="Baseline")

        # LSTM — N_EXPERIMENTS répétitions
        fold_metrics = []
        for exp in range(N_EXPERIMENTS):
            m         = init_model(X_tr, y_tr)
            m, _      = fit_model(m, X_tr, y_tr, verbose=0)
            y_pred    = m.predict(X_te, verbose=0)
            fold_metrics.append(compute_metrics(y_te, y_pred))

        avg = {k: round(np.mean([fm[k] for fm in fold_metrics]), 4) for k in fold_metrics[0]}
        print_metrics(avg, label="LSTM moy.")
        print(f"  🏋🏽 Amélioration MAE : "
              f"{round((1 - avg['MAE'] / metrics_bl['MAE']) * 100, 2)} %")

        results.append({
            'fold': fold_id + 1,
            'baseline_mae':  metrics_bl['MAE'],  'baseline_smape': metrics_bl['SMAPE'],
            'lstm_mae':      avg['MAE'],          'lstm_rmse':      avg['RMSE'],
            'lstm_mape':     avg['MAPE'],         'lstm_smape':     avg['SMAPE'],
            'improvement_%': round((1 - avg['MAE'] / metrics_bl['MAE']) * 100, 2)
        })

    df_results = pd.DataFrame(results)

    print(f"\n{'═'*65}")
    print("RÉCAPITULATIF VALIDATION CROISÉE")
    print(df_results.to_string(index=False))
    print(f"\nMoyenne LSTM SMAPE  : {df_results['lstm_smape'].mean():.2f}%")
    print(f"Amélioration moyenne : {df_results['improvement_%'].mean():.2f}%")

    return df_results


df_results = cross_validate()
