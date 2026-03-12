import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from power_forecast.logic.get_data.download_api import build_feature_dataframe
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, SimpleRNN, Normalization, LSTM
from tensorflow.keras.models import load_model
from pathlib import Path
from datetime import datetime
import random

pd.set_option('display.max_columns', None)

df = build_feature_dataframe('raw_data/all_countries.csv', load_from_pickle=True)


# ── DEFINE INPUT/OUTPUT PARAMETERS ──────────────────────────
 
TARGET_COL       = "FRA"
feature_cols = [c for c in df.columns if c != TARGET_COL]

INPUT_LENGTH     = 7 * 24      # 168h context fed to RNN
OUTPUT_LENGTH    = 24          # predict 24h of target day
HORIZON          = 24          # skip 24h between input end and output
TRAIN_TEST_RATIO = 0.98       # 98% of sequences → train, 2% → test
VAL_RATIO        = 0.1         # 10% of train sequences → validation
SAMPLING_RATIO = 0.7         # sample 70% of possible sequences (for faster training; set to 1.0 to use all)
BATCH_SIZE = 64
EPOCHS = 50

MODEL_NAME    = "lstm"  


model_name = f"{MODEL_NAME}_{TARGET_COL}_in{INPUT_LENGTH}_out{OUTPUT_LENGTH}_h{HORIZON}"


# Paths to save scaler, sequences, and model. Adjust as needed.
SCALER_PATH = Path("raw_data/scalers/scaler.pkl")
SAVE_SEQUENCES = Path("raw_data/sequences")
SAVE_SEQUENCES.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(f"raw_data/models/{model_name}.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

##-------------------- Train - Test split based on input day --------------------

print("Data start:", df.index.min())
print("Data end:  ", df.index.max())
print("Total rows:", len(df))

cutoff = pd.Timestamp("2024-01-01", tz="UTC")

fold_train = df[df.index <  cutoff].copy()
fold_test  = df[df.index >= cutoff - pd.Timedelta(hours=INPUT_LENGTH)].copy()

print(f"\nfold_train: {len(fold_train)} rows  {fold_train.index[0]} → {fold_train.index[-1]}")
print(f"fold_test:  {len(fold_test)} rows   {fold_test.index[0]} → {fold_test.index[-1]}")
print(f"First y_test label: {fold_test.index[INPUT_LENGTH + HORIZON]}")


## ------------------- Standardize features -------------------

# ## STEP 2 — SCALE FEATURES (fit on train only)
# ## ✅ Scaler sees only fold_train → no future information leaks in.
# ## ✅ We scale features only, NOT the target (y).
# ##    Keeping y unscaled makes predictions directly interpretable
# ##    as real electricity prices (€/MWh).

fit_scaler = True  # set to False to skip scaling (useful for debugging)

if fit_scaler:
    scaler = StandardScaler()
    fold_train[feature_cols] = scaler.fit_transform(fold_train[feature_cols])
    joblib.dump(scaler, SCALER_PATH)
else: 
    scaler = joblib.load(SCALER_PATH)


fold_test[feature_cols]  = scaler.transform(fold_test[feature_cols])
## transform only on test — same scaler, no re-fitting



## ------------------- CREATE SEQUENCES (X, y) FOR RNN -------------------

## STEP 3 — SAMPLE SEQUENCES FROM fold_train → 3D arrays
## ✅ Sequences are sampled AFTER scaling (you never want to build
##    sequences from raw data and scale them after — the scaler
##    would not have the right statistics per-sequence).
## ✅ mode='random' for train: model sees diverse starting points.
## Output: X_train (n, 2 weeks data, n_features) | y_train (n, 24)

#Load if they exisits alreaday, otherwise create them and save for future use.
X_train = np.load(SAVE_SEQUENCES / "X_train.npy")
y_train = np.load(SAVE_SEQUENCES / "y_train.npy")
print(f"Loaded — X_train: {X_train.shape}")
print(f"Loaded — y_train: {y_train.shape}")


# Methods to estimate how many sequences can be extracted from a fold,
# and recommend n_sequences based on fold length and sequence span.
# Useful for setting n_sequences_train and n_sequences_test in get_X_y.

def max_sequences(fold: pd.DataFrame) -> int:
    """
    Max number of NON-OVERLAPPING sequences in a fold.
    This is the theoretical ceiling for n_sequences.
    In practice use 50-80% of this to avoid redundancy.
    """
    total_span = INPUT_LENGTH + HORIZON + OUTPUT_LENGTH  
    return max(0, len(fold) - total_span)

def recommend_n_sequences(fold_train, fold_test):
    max_train = max_sequences(fold_train)
    max_test  = max_sequences(fold_test)
    
    print(f"fold_train length : {len(fold_train)} rows")
    print(f"fold_test  length : {len(fold_test)} rows")
    print(f"One sequence span : {INPUT_LENGTH + HORIZON + OUTPUT_LENGTH} rows")
    print(f"Max train sequences (non-overlapping): {max_train}")
    print(f"Max test  sequences (non-overlapping): {max_test}")
    print(f"→ Recommended n_sequences_train: {int(max_train * 0.7)}")
    print(f"→ Recommended n_sequences_test:  {int(max_test  * 0.7)}")


n_sequences_train = int(max_sequences(fold_train) * SAMPLING_RATIO)

# If sequenced X_train and y_train do not exist, create them and save for future use.
resample_sequences = True  # set to True to re-sample sequences from fold_train
print(f"If you want to re-sample, it will create {n_sequences_train} train sequences.")

# Method to transform our X_train in 3-dimensional array of sequences, and y_train in 2D array of corresponding labels.


def get_Xi_yi(fold: pd.DataFrame, feature_cols: list, target_col: str,
              input_length: int, horizon: int, output_length: int,
              start_idx: int = None):
    """
    Extract one (X_i, y_i) sequence pair from a fold.

    Sequence layout:
        [start_idx ──── start+input_length)     → X_i (features fed to RNN)
        [start+input_length ── +horizon)         → skipped (unknown future)
        [start+input_length+horizon ── +output)  → y_i (prices to predict)

    start_idx=None → random sampling
    start_idx=int  → deterministic (used for chronological scan)
    """
    total_span = input_length + horizon + output_length
    max_start  = len(fold) - total_span

    if max_start <= 0:
        raise ValueError(
            f"Fold too short ({len(fold)} rows) for one sequence ({total_span} rows). "
            f"Reduce INPUT_LENGTH or use a longer fold."
        )

    if start_idx is None:
        start_idx = np.random.randint(0, max_start + 1)

    X_i = fold[feature_cols].iloc[start_idx : start_idx + input_length].values
    # shape → (input_length, n_features)

    y_start = start_idx + input_length + horizon
    y_i = fold[target_col].iloc[y_start : y_start + output_length].values
    # shape → (output_length,)

    return X_i, y_i


def get_X_y(fold: pd.DataFrame, feature_cols: list, target_col: str,
            input_length: int, horizon: int, output_length: int,
            n_sequences: int = None, mode: str = 'random'):
    """
    Build (X, y) 3D arrays from a fold.

    mode='random':        randomly sample n_sequences pairs (with possible overlap)
                          → good for TRAIN (model sees diverse starting points)
    mode='chronological': scan the fold step by step, all possible pairs
                          → good for TEST (no randomness, deterministic evaluation)

    Returns:
        X: (n_sequences, input_length, n_features)
        y: (n_sequences, output_length)
    """
    total_span = input_length + horizon + output_length
    max_start  = len(fold) - total_span

    if max_start <= 0:
        raise ValueError(f"Fold too short: {len(fold)} rows, need {total_span}")

    if mode == 'chronological':
        indices = range(0, max_start + 1)  # every possible start
    else:
        if n_sequences is None:
            raise ValueError("n_sequences required for random mode")
        indices = [None] * n_sequences     # None → random in get_Xi_yi

    X_list, y_list = [], []
    for idx in indices:
        X_i, y_i = get_Xi_yi(fold, feature_cols, target_col,
                              input_length, horizon, output_length,
                              start_idx=idx)
        X_list.append(X_i)
        y_list.append(y_i)

    return np.array(X_list), np.array(y_list)



if resample_sequences:
    X_train, y_train = get_X_y(fold_train, feature_cols, TARGET_COL,
                              INPUT_LENGTH, HORIZON, OUTPUT_LENGTH,
                              n_sequences=n_sequences_train, mode='random')
    np.save(SAVE_SEQUENCES / "X_train.npy", X_train)
    np.save(SAVE_SEQUENCES / "y_train.npy", y_train)
    print(f"Sampled and saved — X_train: {X_train.shape}")
    print(f"Sampled and saved — y_train: {y_train.shape}")




#-------------- Validation split from train sequences --------------

print(f"Loaded — X_train: {X_train.shape}")
print(f"Loaded — y_train: {y_train.shape}")

val_split  = int(len(X_train) * (1 - VAL_RATIO))

X_tr  = X_train[:val_split]
y_tr  = y_train[:val_split]
X_val = X_train[val_split:]
y_val = y_train[val_split:]

print(f"\nX_tr:  {X_tr.shape}   y_tr:  {y_tr.shape}")
print(f"X_val: {X_val.shape}  y_val: {y_val.shape}")


#------------------- CREATE AND TRAIN RNN -------------------

# Variable to see if we want to train a new model or load an existing one.
train_new_model = True  # set to True to skip training and load existing model
# model_name = f"{MODEL_NAME}_{TARGET_COL}_in{INPUT_LENGTH}_out{OUTPUT_LENGTH}_h{HORIZON}"
MODEL_NAME    = "lstm"  
BATCH_SIZE = 64
EPOCHS = 50


# ─────────────────────────────────────────────────────────────────
## LSTM MODEL
# ─────────────────────────────────────────────────────────────────

def initialize_model_lstm(input_shape, output_length):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, activation='tanh', return_sequences=True),  # return full sequence to next LSTM
        LSTM(32, activation='tanh', return_sequences=False), # compress to final hidden state
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(output_length, activation='linear')
    ])

    model.compile(
        optimizer='adam',
        loss='mae',
        metrics=['mae', 'mse']
    )
    model.summary()
    return model


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

def train_or_load_model_lstm(train_new_model, X_tr, y_tr, X_val, y_val,
                        OUTPUT_LENGTH, EPOCHS, BATCH_SIZE,
                        early_stopping, MODEL_PATH):

    if train_new_model:
        model = initialize_model_lstm(input_shape=X_tr.shape[1:], output_length=OUTPUT_LENGTH)

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=1
        )

        model.save(MODEL_PATH)
        print(f"Trained and saved model to {MODEL_PATH}")

    else:
        print(f"Skipping training, using existing model at {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        history = None

    model.summary()
    return model, history


model_lstm, history_lstm = train_or_load_model_lstm(
    train_new_model,
    X_tr, y_tr,
    X_val, y_val,
    OUTPUT_LENGTH,
    EPOCHS,
    BATCH_SIZE,
    early_stopping,
    MODEL_PATH
)


# ---- Create X_test and y_test from fold_test (chronological scan) ----
X_test, y_test = get_X_y(
    fold=fold_test,
    feature_cols=feature_cols,
    target_col=TARGET_COL,
    input_length=INPUT_LENGTH,
    horizon=HORIZON,
    output_length=OUTPUT_LENGTH,
    mode="chronological"
)

# Evaluate model on test set
loss, mae, mse = model_lstm.evaluate(X_test, y_test, verbose=1)


print("Test Loss:", loss)
print("Test MAE:", mae)
print("Test MSE:", mse)
