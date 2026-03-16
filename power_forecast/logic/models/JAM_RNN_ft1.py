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
"""

import pandas as pd
from power_forecast.logic.get_data.build_dataframe import build_feature_dataframe
# %load_ext autoreload
# %autoreload 2
pd.set_option('display.max_columns', None)

# Chargement du DataFrame de features à partir d'un CSV multi-pays.
# `load_from_pickle=False` force un rechargement complet plutôt que d'utiliser un cache.
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
from keras.callbacks import EarlyStopping

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

df_selected = df[selected_features + ['FRA']]
print(f"\nShape après feature selection : {df_selected.shape}")

# --------------------------------------------------- #
# Configuration globale du jeu de données             #
# --------------------------------------------------- #

TARGET          = 'FRA'
N_FEATURES      = df_selected.shape[1]

FOLD_LENGTH      = 24 * 365 * 2   # 2 ans (~17 520h, proche des 17 000h de [2])
FOLD_STRIDE      = 24 * 91        # 1 trimestre entre chaque fold
TRAIN_TEST_RATIO = 0.7

# INPUT_LENGTH = 96h — fenêtre physiquement motivée par [2] (section IV) :
# à 4 jours, les prix sont gaussiens.
INPUT_LENGTH    = 96    # 4 jours
OUTPUT_LENGTH   = 1
SEQUENCE_STRIDE = 24    # 1 pas/jour
DAY_AHEAD_GAP   = 24    # Prévision J+1


print(f"N_FEATURES = {N_FEATURES} | INPUT_LENGTH = {INPUT_LENGTH}h = {INPUT_LENGTH//24} jours")


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

    Exemple :
        Avec fold_length=100 et fold_stride=50, les folds démarrent aux indices 0, 50, 100, ...
    """
    folds = []
    for idx in range(0, len(df), fold_stride):
        # On s'arrête dès que la fenêtre dépasserait la fin du DataFrame
        if (idx + fold_length) > len(df):
            break
        fold = df.iloc[idx:idx + fold_length, :]
        folds.append(fold)
    return folds


folds = get_folds(df, FOLD_LENGTH, FOLD_STRIDE)

print(f'The function generated {len(folds)} folds.')
print(f'Each fold has a shape equal to {folds[0].shape}.')

fold=folds[0]
fold



def train_test_split(fold: pd.DataFrame,
                     train_test_ratio: float,
                     input_length: int) -> Tuple[pd.DataFrame]:
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
        Tuple[pd.DataFrame]: (fold_train, fold_test)
            - fold_train : lignes [0, last_train_idx)
            - fold_test  : lignes [last_train_idx - input_length, fin)
    """
    # ENSEMBLE D'ENTRAÎNEMENT
    last_train_idx = round(train_test_ratio * len(fold))
    fold_train = fold.iloc[0:last_train_idx, :]

    # ENSEMBLE DE TEST — chevauchement volontaire pour avoir une première fenêtre complète
    first_test_idx = last_train_idx - input_length
    fold_test = fold.iloc[first_test_idx:, :]

    return (fold_train, fold_test)


(fold_train, fold_test) = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)

# Résumé des dimensions des séquences
print(f'N_FEATURES = {N_FEATURES}')
print(f'INPUT_LENGTH = {INPUT_LENGTH} timesteps = {int(INPUT_LENGTH)/24} days = {int(INPUT_LENGTH/24/7)} weeks')



# On prédit exactement 1 valeur, 24h après la fin de la fenêtre d'entrée
OUTPUT_LENGTH = 1
print(f'OUTPUT_LENGTH = {OUTPUT_LENGTH}')


def get_Xi_yi(
    fold: pd.DataFrame,
    input_length: int,
    output_length: int) -> Tuple[pd.DataFrame]:
    """
    Extrait une paire (entrée, cible) depuis un fold en choisissant un point de départ aléatoire.

    La cible `y_i` est positionnée 24 heures après la fin de `X_i`, simulant un
    scénario de prévision à horizon J+1 (day-ahead forecasting).

    Args:
        fold (pd.DataFrame): Un seul fold de la série temporelle.
        input_length (int): Nombre de pas de temps dans la séquence d'entrée X_i.
        output_length (int): Nombre de pas de temps à prédire (y_i).

    Returns:
        Tuple[pd.DataFrame]: (X_i, y_i)
            - X_i : forme (input_length, n_features)
            - y_i : forme (output_length, 1) — uniquement la colonne TARGET
    """
    first_possible_start = 0
    last_possible_start = len(fold) - (input_length + output_length + 24) + 1
    random_start = np.random.randint(first_possible_start, last_possible_start)

    X_i = fold.iloc[random_start:random_start + input_length]
    # On saute 24h après la fin de la fenêtre d'entrée (décalage day-ahead)
    y_i = fold.iloc[random_start + input_length + 24:
                    random_start + input_length + output_length + 24][[TARGET]]

    return (X_i, y_i)


X_train_i, y_train_i = get_Xi_yi(fold_train, INPUT_LENGTH, OUTPUT_LENGTH)
X_test_i, y_test_i = get_Xi_yi(fold_test, INPUT_LENGTH, OUTPUT_LENGTH)

# Vérification : la dernière séquence possible doit se terminer sur la dernière ligne de fold_test
X_last, y_last = get_Xi_yi(fold_test, input_length=(len(fold_test) - OUTPUT_LENGTH - 24), output_length=OUTPUT_LENGTH)
assert y_last.values == fold_test.iloc[-1, :][TARGET]


def get_X_y(
    fold: pd.DataFrame,
    number_of_sequences: int,
    input_length: int,
    output_length: int) -> Tuple[np.array]:
    """
    Construit un jeu de données en échantillonnant aléatoirement des paires (X_i, y_i) dans un fold.

    Appelle `get_Xi_yi` de manière répétée pour créer `number_of_sequences` échantillons indépendants.
    L'échantillonnage aléatoire introduit de la diversité dans l'ensemble d'entraînement.

    Args:
        fold (pd.DataFrame): Le fold depuis lequel échantillonner.
        number_of_sequences (int): Nombre total de paires (X, y) à générer.
        input_length (int): Longueur de chaque séquence d'entrée.
        output_length (int): Longueur de chaque séquence cible.

    Returns:
        Tuple[np.array]: (X, y)
            - X : forme (number_of_sequences, input_length, n_features)
            - y : forme (number_of_sequences, output_length, 1)
    """
    X, y = [], []
    for i in range(number_of_sequences):
        (Xi, yi) = get_Xi_yi(fold, input_length, output_length)
        X.append(Xi)
        y.append(yi)
    return np.array(X), np.array(y)


N_TRAIN = 1000  # Nombre de séquences d'entraînement générées aléatoirement
N_TEST = 100   # Nombre de séquences de test générées aléatoirement

X_train, y_train = get_X_y(fold_train, N_TRAIN, INPUT_LENGTH, OUTPUT_LENGTH)
X_test, y_test = get_X_y(fold_test, N_TEST, INPUT_LENGTH, OUTPUT_LENGTH)

# SEQUENCE_STRIDE : décalage entre deux séquences consécutives dans l'approche par pas fixe
SEQUENCE_STRIDE = 24  # On avance d'1 jour entre chaque séquence


def get_X_y_strides(fold: pd.DataFrame, input_length: int, output_length: int,
                    sequence_stride: int) -> Tuple[np.array]:
    """
    Construit un jeu de données en faisant glisser une fenêtre à pas fixe sur le fold.

    Contrairement à `get_X_y` (qui échantillonne aléatoirement), cette fonction parcourt
    le fold de manière déterministe. Elle couvre ainsi l'intégralité du fold sans trou
    et sans risque de fuite de données entre séquences.

    Note : il n'y a PAS de décalage de 24h ici (contrairement à `get_Xi_yi`) — y_i suit
    immédiatement X_i. Cette variante est utilisée pour le pipeline d'entraînement principal.

    Args:
        fold (pd.DataFrame): Un seul fold de la série temporelle.
        input_length (int): Nombre de pas de temps dans chaque fenêtre X_i.
        output_length (int): Nombre de pas de temps dans chaque cible y_i.
        sequence_stride (int): Nombre de pas de temps entre le début de deux séquences consécutives.

    Returns:
        Tuple[np.array]: (X, y)
            - X : forme (n_sequences, input_length, n_features)
            - y : forme (n_sequences, output_length, 1)
    """
    X, y = [], []
    for i in range(0, len(fold), sequence_stride):
        # On s'arrête si la fenêtre (X + y) dépasse la longueur du fold
        if (i + input_length + output_length) >= len(fold):
            break
        X_i = fold.iloc[i:i + input_length, :]
        y_i = fold.iloc[i + input_length:i + input_length + output_length, :][[TARGET]]
        X.append(X_i)
        y.append(y_i)
    return (np.array(X), np.array(y))


print("FOLD_LENGTH")
print(f"= {FOLD_LENGTH} timesteps")
print(f"= {int(FOLD_LENGTH/24)} days")
print(f"= {int(FOLD_LENGTH/24/7)} weeks")

X_train, y_train = get_X_y_strides(fold_train, INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)
X_test, y_test = get_X_y_strides(fold_test, INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

print(X_train.shape)
print(y_train.shape)


def init_model(X_train: np.array, y_train: np.array) -> tf.keras.Model:
    """
    Construit et compile le modèle LSTM pour la prévision de série temporelle.

    Architecture :
        1. Couche de normalisation — standardise chaque feature selon les statistiques d'entraînement.
        2. Couche LSTM (64 unités) — capture les dépendances temporelles sur la fenêtre d'entrée.
           Une régularisation L1L2 est appliquée pour limiter le surapprentissage.
        3. Couche Dense de sortie — projette l'état caché LSTM sur `output_length` prédictions.

    Compilation :
        - Fonction de perte : Erreur Quadratique Moyenne (MSE)
        - Optimiseur : Adam (lr=0.005)
        - Métrique : Erreur Absolue Moyenne (MAE)

    Args:
        X_train (np.array): Entrées d'entraînement de forme (n_samples, input_length, n_features).
        y_train (np.array): Cibles d'entraînement de forme (n_samples, output_length, 1).

    Returns:
        tf.keras.Model : Modèle Keras Sequential compilé, prêt pour l'entraînement.
    """
    # Adaptation de la couche de normalisation sur les statistiques des données d'entraînement
    normalizer = Normalization()
    normalizer.adapt(X_train)

    model = models.Sequential()
    model.add(Input(shape=X_train[0].shape))
    model.add(normalizer)                          # Standardisation des features
    model.add(layers.LSTM(
        64,
        activation='tanh',
        return_sequences=False,                    # On ne retourne que le dernier état caché
        kernel_regularizer=L1L2(l1=0.05, l2=0.05) # Régularisation pour limiter le surapprentissage
    ))
    output_length = y_train.shape[1]
    model.add(layers.Dense(output_length, activation='linear'))  # Sortie linéaire pour la régression

    adam = optimizers.Adam(learning_rate=0.005)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model


model = init_model(X_train, y_train)
model.summary()


def plot_history(history: tf.keras.callbacks.History):
    """
    Affiche les courbes de perte (MSE) et de métrique (MAE) pour l'entraînement et la validation.

    Affiche deux sous-graphiques côte à côte :
        - Gauche  : courbes de MSE (train et validation) en fonction des époques.
        - Droite  : courbes de MAE (train et validation) en fonction des époques.

    Utile pour détecter le surapprentissage (train s'améliore mais validation diverge)
    ou le sous-apprentissage (les deux courbes plafonnent à des valeurs élevées).

    Args:
        history (tf.keras.callbacks.History): L'objet History retourné par `model.fit()`.

    Returns:
        np.array : Tableau de deux objets Axes matplotlib.
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))

    # Courbe de perte MSE
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('MSE')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='best')
    ax[0].grid(axis="x", linewidth=0.5)
    ax[0].grid(axis="y", linewidth=0.5)

    # Courbe de métrique MAE
    ax[1].plot(history.history['mae'])
    ax[1].plot(history.history['val_mae'])
    ax[1].set_title('MAE')
    ax[1].set_ylabel('MAE')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='best')
    ax[1].grid(axis="x", linewidth=0.5)
    ax[1].grid(axis="y", linewidth=0.5)

    return ax


def fit_model(model: tf.keras.Model, verbose: int = 1) -> Tuple[tf.keras.Model, dict]:
    """
    Entraîne le modèle LSTM avec arrêt anticipé (early stopping) sur la perte de validation.

    Configuration d'entraînement :
        - 30% des données d'entraînement sont mises de côté comme ensemble de validation.
        - Les séquences NE sont PAS mélangées pour préserver l'ordre temporel.
        - L'early stopping surveille `val_loss` avec une patience de 10 époques
          et restaure les meilleurs poids observés pendant l'entraînement.

    Args:
        model (tf.keras.Model): Un modèle Keras compilé (issu de `init_model`).
        verbose (int): Niveau de verbosité pour `model.fit()` (0=silencieux, 1=barre de progression).

    Returns:
        Tuple[tf.keras.Model, dict]:
            - model   : Le modèle entraîné avec les meilleurs poids restaurés.
            - history : L'objet History contenant les pertes/métriques par époque.
    """
    es = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        shuffle=False,           # Critique : ne pas mélanger les données de série temporelle
        batch_size=16,
        epochs=100,
        callbacks=[es],
        verbose=verbose
    )

    return model, history


# ====================================
# 1 - Initialisation et résumé
# ====================================
model = init_model(X_train, y_train)
model.summary()

# ====================================
# 2 - Entraînement
# ====================================
model, history = fit_model(model)
plot_history(history)

# ====================================
# 3 - Évaluation sur l'ensemble de test
# ====================================
res = model.evaluate(X_test, y_test)
print(f"The LSTM MAE on the test set is equal to {round(res[1], 2)} Euro")


def init_baseline() -> tf.keras.Model:
    """
    Construit et compile un modèle baseline naïf de type "dernière valeur observée".

    Le baseline prédit la valeur TARGET en renvoyant simplement la dernière valeur
    observée de cette feature dans la séquence d'entrée (x[:, -1, 1]).
    Cela simule l'hypothèse naïve "demain ressemblera à aujourd'hui".

    Ce modèle sert de borne inférieure de performance : si le LSTM ne fait pas mieux,
    c'est qu'il n'a rien appris d'utile.

    Returns:
        tf.keras.Model : Modèle Keras compilé qui retourne la dernière valeur observée de TARGET.
    """
    model = models.Sequential()
    # La couche Lambda extrait la valeur du dernier pas de temps pour la feature d'index 1 (TARGET)
    model.add(layers.Lambda(lambda x: x[:, -1, 1, None]))

    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model


baseline_model = init_baseline()
baseline_score = baseline_model.evaluate(X_test, y_test)
print(f"- The Baseline MAE on the test set is equal to {round(baseline_score[1], 2)} Euros")

print(f"- The LSTM MAE on the test set is equal to {round(res[1], 2)} Euros")
print(f"🔥 Improvement of the LSTM model over the baseline (on this fold for the test set) = : {round((1 - (res[1] / baseline_score[1])) * 100, 2)} % 🔥")

# Récapitulatif de tous les hyperparamètres globaux pour la reproductibilité
print(f'N_FEATURES = {N_FEATURES}')
print('')
print(f'FOLD_LENGTH = {FOLD_LENGTH}')
print(f'FOLD_STRIDE = {FOLD_STRIDE}')
print(f'TRAIN_TEST_RATIO = {TRAIN_TEST_RATIO}')
print('')
print(f'N_TRAIN = {N_TRAIN}')
print(f'N_TEST = {N_TEST}')
print(f'INPUT_LENGTH = {INPUT_LENGTH}')
print(f'OUTPUT_LENGTH = {OUTPUT_LENGTH}')


def cross_validate_baseline_and_lstm():
    """
    Effectue une validation croisée temporelle en comparant le baseline et le modèle LSTM.

    Pour chaque fold généré depuis le jeu de données complet :
        1. Division en ensembles train et test.
        2. Évaluation du baseline naïf (dernière valeur observée).
        3. Entraînement d'un modèle LSTM et évaluation sur le même ensemble de test.
        4. Affichage de la MAE des deux modèles et du gain en pourcentage.

    Cette approche mesure la régularité avec laquelle le LSTM surpasse le baseline sur
    différentes périodes, offrant une estimation de performance plus robuste qu'une
    simple séparation train/test.

    Returns:
        Tuple[List[float], List[float]]:
            - list_of_mae_baseline_model  : MAE du baseline pour chaque fold.
            - list_of_mae_recurrent_model : MAE du LSTM pour chaque fold.
    """
    list_of_mae_baseline_model = []
    list_of_mae_recurrent_model = []

    # Génération de tous les folds depuis le jeu de données complet
    folds = get_folds(df, FOLD_LENGTH, FOLD_STRIDE)

    for fold_id, fold in enumerate(folds):

        # 1 - Division du fold en train et test
        (fold_train, fold_test) = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)

        X_train, y_train = get_X_y(fold_train, N_TRAIN, INPUT_LENGTH, OUTPUT_LENGTH)
        X_test, y_test = get_X_y(fold_test, N_TEST, INPUT_LENGTH, OUTPUT_LENGTH)

        # 2a - Baseline : évaluation sans entraînement
        baseline_model = init_baseline()
        mae_baseline = baseline_model.evaluate(X_test, y_test, verbose=0)[1]
        list_of_mae_baseline_model.append(mae_baseline)
        print("-" * 50)
        print(f"MAE baseline fold n°{fold_id} = {round(mae_baseline, 2)}")

        # 2b - LSTM : entraînement avec early stopping, puis évaluation
        model = init_model(X_train, y_train)
        es = EarlyStopping(
            monitor="val_mae",
            mode="min",
            patience=2,
            restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            shuffle=False,
            batch_size=16,
            epochs=100,
            callbacks=[es],
            verbose=0
        )
        res = model.evaluate(X_test, y_test, verbose=0)
        mae_lstm = res[1]
        list_of_mae_recurrent_model.append(mae_lstm)
        print(f"MAE LSTM fold n°{fold_id} = {round(mae_lstm, 2)}")

        # Comparaison LSTM vs baseline pour ce fold
        print(f"🏋🏽‍♂️ improvement over baseline: {round((1 - (mae_lstm / mae_baseline)) * 100, 2)} % \n")

    return list_of_mae_baseline_model, list_of_mae_recurrent_model


mae_baselines, mae_lstms = cross_validate_baseline_and_lstm()
