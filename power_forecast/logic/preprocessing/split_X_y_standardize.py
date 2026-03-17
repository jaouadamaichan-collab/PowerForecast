import numpy as np
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




# ── Core: single sequence ──────────────────────────────────────────────────
def get_Xi_yi_single_sequence(
    fold: pd.DataFrame,
    feature_cols: list,
    country_objective: str,
    start_idx: int,
    input_length: int = INPUT_LENGTH,
    output_length: int = OUTPUT_LENGTH,
) -> tuple:
    """
    Extrait une paire (X_i, y_i) depuis fold à partir de start_idx.

        [start_idx → start_idx + input_length)              → X_i  (input_length, n_features)
        [start_idx + input_length → + output_length)        → y_i  (output_length,)

    y commence IMMÉDIATEMENT après la fin de X (pas de horizon).
    """
    target_col = VILLE_TO_ISO.get(country_objective, None)
    X_i = fold[feature_cols].iloc[start_idx : start_idx + input_length].values
    y_start = start_idx + input_length
    y_i = fold[target_col].iloc[y_start : y_start + output_length].values
    return X_i, y_i

# ── Vectorized: all sequences at once ─────────────────────────────────────
def get_X_y_vectorized_RNN(
    fold: pd.DataFrame,
    feature_cols: list,
    country_objective: str,
    stride: int,
    input_length: int = INPUT_LENGTH,
    output_length: int = OUTPUT_LENGTH,
    scaler=None,       # StandardScaler fitté sur X_train, ou None
    fit_scaler=False,  # True uniquement pour le fold train
) -> tuple:
    """
    Construit toutes les séquences (X, y) par fenêtre glissante vectorisée.

    ⚠️  Règle anti-leakage : stride >= output_length est obligatoire.
        Si stride < output_length, des séquences y consécutives se chevauchent
        → le modèle prédit des valeurs déjà vues dans le y voisin.

    Schéma d'une fenêtre :
        |<-------- input_length ------->|<--- output_length --->|
        t=start                       t=start+input_length     t=start+input_length+output_length

    Args:
        fold          : DataFrame du fold (train ou test).
        feature_cols  : Colonnes utilisées comme features pour X.
        target_col    : Colonne cible pour y.
        stride        : Pas entre deux séquences. Doit être >= output_length.
        input_length  : Longueur de chaque séquence X.
        output_length : Longueur de chaque séquence y.

    Returns:
        X : np.ndarray de shape (n_seq, input_length, n_features)
        y : np.ndarray de shape (n_seq, output_length)
    """
    # ── garde-fou leakage ──────────────────────────────────────────────────
    
    if stride < output_length:
        raise ValueError(
            f"⛔ Leakage détecté : stride={stride} < output_length={output_length}. "
            f"Les séquences y se chevaucheraient. Utilise stride >= {output_length}."
        )

    total_span = input_length + output_length

    if len(fold) < total_span:
        raise ValueError(
            f"Fold trop court : {len(fold)} lignes, minimum requis = {total_span}."
        )

    target_col = VILLE_TO_ISO.get(country_objective, None)
    
    X_all = fold[feature_cols].values  # (n_rows, n_features)
    y_all = fold[target_col].values    # (n_rows,)

    # toutes les fenêtres possibles de taille total_span → (n_windows, total_span, n_features)
    X_wins = np.lib.stride_tricks.sliding_window_view(
        X_all, window_shape=(total_span, X_all.shape[1])
    )[:, 0, :, :]  # (n_windows, total_span, n_features)

    # sous-échantillonnage par stride
    X_wins = X_wins[::stride]  # (n_seq, total_span, n_features)

    # extraction X et y
    X = X_wins[:, :input_length, :].copy()   # (n_seq, input_length, n_features), .copy()  # ← sécurise la contiguïté mémoire

    y_wins = np.lib.stride_tricks.sliding_window_view(y_all, total_span)[::stride]
    y = y_wins[:, input_length:]      # (n_seq, output_length)

    # ── standardisation de X uniquement ───────────────────────────────────
    if scaler is not None:
        n_seq, in_len, n_feat = X.shape
        X_flat = X.reshape(-1, n_feat)          # (n_seq * input_length, n_features)

        if fit_scaler:
            X_flat = scaler.fit_transform(X_flat)
        else:
            X_flat = scaler.transform(X_flat)

        X = X_flat.reshape(n_seq, in_len, n_feat)
        
    print(
        f"  → {len(X)} séquences générées  "
        f"(fold={len(fold)}h, stride={stride}h, span={total_span}h, "
        f"leakage-free={'✅' if stride >= output_length else '❌'})"
    )

    return X, y