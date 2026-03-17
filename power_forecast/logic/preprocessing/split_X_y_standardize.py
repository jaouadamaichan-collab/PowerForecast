import pandas as pd
from sklearn.preprocessing import StandardScaler
from power_forecast.params import *

def X_y_standardizer_XGB(fold_train: pd.DataFrame, fold_test: pd.DataFrame, country_objective: str):
    """
    Extrait X/y depuis les folds train et test, puis standardise les features.

    Paramètres
    ----------
    fold_train : pd.DataFrame  — données d'entraînement (timeseries)
    fold_test  : pd.DataFrame  — données de test (timeseries)
    country_objective : str    — nom du pays cible (clé dans VILLE_TO_ISO)

    Retourne
    --------
    X_train_scaled, X_test_scaled, y_train, y_test
    """
    iso_objective = VILLE_TO_ISO.get(country_objective, None)

    y_train = fold_train[iso_objective]
    y_test  = fold_test[iso_objective]

    X_train = fold_train.drop(columns=[iso_objective])
    X_test  = fold_test.drop(columns=[iso_objective])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def X_y_standardizer_with_val_XGB(
    fold_train: pd.DataFrame,
    fold_test: pd.DataFrame,
    country_objective: str,
    val_ratio: float = 0.2,
):
    """
    Identique à X_y_standardizer mais découpe chronologiquement X_train
    en un jeu d'entraînement réduit et un jeu de validation.

    Paramètres
    ----------
    fold_train       : pd.DataFrame — données d'entraînement (timeseries)
    fold_test        : pd.DataFrame — données de test (timeseries)
    country_objective: str          — nom du pays cible (clé dans VILLE_TO_ISO)
    val_ratio        : float        — proportion de fin de train utilisée pour val (défaut 0.2)

    Retourne
    --------
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    """
    iso_objective = VILLE_TO_ISO.get(country_objective, None)

    y_full_train = fold_train[iso_objective]
    y_test       = fold_test[iso_objective]

    X_full_train = fold_train.drop(columns=[iso_objective])
    X_test       = fold_test.drop(columns=[iso_objective])

    # Coupure chronologique (pas de shuffle)
    split_idx = int(len(X_full_train) * (1 - val_ratio))

    X_train = X_full_train.iloc[:split_idx]
    X_val   = X_full_train.iloc[split_idx:]
    y_train = y_full_train.iloc[:split_idx]
    y_val   = y_full_train.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test