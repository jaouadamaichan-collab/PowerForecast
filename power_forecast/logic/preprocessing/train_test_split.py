import pandas as pd

def train_test_split_general(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Basic train/test split based on a cutoff date.

    Parameters
    ----------
    df      : pd.DataFrame with DatetimeIndex (UTC)
    cutoff  : pd.Timestamp, split date (tz-aware)

    Returns
    -------
    fold_train : df[index < cutoff]
    fold_test  : df[index >= cutoff]
    """
    fold_train = df[df.index < cutoff].copy()
    fold_test  = df[df.index >= cutoff].copy()

    print(f"fold_train: {len(fold_train)} rows  {fold_train.index[0]} → {fold_train.index[-1]}")
    print(f"fold_test:  {len(fold_test)} rows   {fold_test.index[0]} → {fold_test.index[-1]}")

    return fold_train, fold_test


# ⚠️ Rappel important sur l'évaluation des modèles
# Les fonctions ci-dessous effectuent un split unique basé sur un jour objectif choisi manuellement.
# Si ce jour correspond à une période atypique (pic de prix, crise énergétique, jours fériés...),
# les métriques obtenues ne seront pas représentatives des performances réelles du modèle.
# Pour un projet de démo c'est suffisant, mais ne pas tirer de conclusions définitives sur la qualité
# du modèle sans avoir testé sur plusieurs périodes différentes (walk-forward validation).

def train_test_split_XGB_optimized(
    df: pd.DataFrame,
    objective_day: pd.Timestamp,
    number_days_to_predict: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/test split optimized for XGBoost.

    Train : from df start → objective_day (exclusive)
    Test  : from objective_day → objective_day + number_days_to_predict

    No INPUT_LENGTH overlap needed since XGB uses explicit lag columns
    already baked into the features — no raw history window required at inference.

    Parameters
    ----------
    df                     : pd.DataFrame with DatetimeIndex (UTC)
    objective_day          : pd.Timestamp, first day to predict (tz-aware)
    number_days_to_predict : int, number of days to include in test set

    Returns
    -------
    fold_train, fold_test
    """
    test_end = objective_day + pd.Timedelta(days=number_days_to_predict)

    fold_train = df[df.index < objective_day].copy()
    fold_test  = df[(df.index >= objective_day) & (df.index < test_end)].copy()

    print(f"    fold_train: {len(fold_train)} rows  {fold_train.index[0]} → {fold_train.index[-1]}")
    print(f"    fold_test:  {len(fold_test)} rows   {fold_test.index[0]} → {fold_test.index[-1]}")

    return fold_train, fold_test


def train_test_split_RNN_optimized(
    df: pd.DataFrame,
    objective_day: pd.Timestamp,
    number_days_to_predict: int,
    input_length: int,  # in hours
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/test split optimized for RNN/LSTM.

    Train : from df start → objective_day - input_length hours (exclusive)
            The gap avoids any overlap between the last training sequence
            and the test window, preventing data leakage.
    Test  : from (objective_day - input_length hours) → objective_day + number_days_to_predict
            The test set includes the INPUT_LENGTH lookback rows the model
            needs to build its first input sequence at inference time.

    Parameters
    ----------
    df                     : pd.DataFrame with DatetimeIndex (UTC)
    objective_day          : pd.Timestamp, first day to predict (tz-aware)
    number_days_to_predict : int, number of days to include in test set
    input_length           : int, RNN lookback window in hours (e.g. 7*24 = 168)

    Returns
    -------
    fold_train, fold_test
    """
    train_cutoff = objective_day - pd.Timedelta(hours=input_length)
    test_end     = objective_day + pd.Timedelta(days=number_days_to_predict)

    fold_train = df[df.index < train_cutoff].copy()
    fold_test  = df[(df.index >= train_cutoff) & (df.index < test_end)].copy()

    print(f"fold_train: {len(fold_train)} rows  {fold_train.index[0]} → {fold_train.index[-1]}")
    print(f"fold_test:  {len(fold_test)} rows   {fold_test.index[0]} → {fold_test.index[-1]}")

    return fold_train, fold_test