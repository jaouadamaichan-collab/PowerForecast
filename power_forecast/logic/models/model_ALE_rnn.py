import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from power_forecast.logic.get_data.build_dataframe import build_feature_dataframe
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    Dense,
    SimpleRNN,
    Normalization,
    LSTM,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from datetime import datetime
import random

pd.set_option("display.max_columns", None)

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

# ── DEFINE INPUT/OUTPUT PARAMETERS ──────────────────────────

TARGET_COL = "FRA"
feature_cols = [c for c in df.columns]

INPUT_LENGTH = 14 * 24  # 168h context fed to RNN
OUTPUT_LENGTH = 48  # predict 24h of target day
HORIZON = 0  # skip 24h between input end and output
TRAIN_TEST_RATIO = 0.98  # 98% of sequences → train, 2% → test
VAL_RATIO = 0.1  # 10% of train sequences → validation
STRIDE_TRAIN = 48  # advance 2 day between train sequences
STRIDE_TEST = 48  # advance 2 day between test sequences (deterministic)


# sample 70% of possible sequences (for faster training; set to 1.0 to use all)
PATIENCE = 10
BATCH_SIZE = 64
EPOCHS = 100
MODEL_NAME = "lstm"

fit_scaler = True  # set to False to skip scaling (useful for debugging)
resample_sequences = False
train_new_model = False  # set to True to skip training and load existing model


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

cutoff = pd.Timestamp("2023-10-01", tz="UTC")

fold_train = df[df.index < cutoff].copy()
fold_test = df[df.index >= cutoff - pd.Timedelta(hours=INPUT_LENGTH)].copy()

print(
    f"\nfold_train: {len(fold_train)} rows  {fold_train.index[0]} → {fold_train.index[-1]}"
)
print(
    f"fold_test:  {len(fold_test)} rows   {fold_test.index[0]} → {fold_test.index[-1]}"
)
print(f"First y_test label: {fold_test.index[INPUT_LENGTH + HORIZON]}")


# Scale before sampling sequences, also the target column
# It will be scaled back to real values at the end for evaluation and plotting.
scaler = StandardScaler()
fold_train[feature_cols] = scaler.fit_transform(fold_train[feature_cols])
fold_test[feature_cols] = scaler.transform(fold_test[feature_cols])


## ------------------- CREATE SEQUENCES (X, y) FOR RNN -------------------

## STEP 3 — SAMPLE SEQUENCES FROM fold_train → 3D arrays
## ✅ Sequences are sampled AFTER scaling (you never want to build
##    sequences from raw data and scale them after — the scaler
##    would not have the right statistics per-sequence).
## ✅ mode='random' for train: model sees diverse starting points.
## Output: X_train (n, 2 weeks data, n_features) | y_train (n, 24)

# Load if they exisits alreaday, otherwise create them and save for future use.
if resample_sequences == False:
    X_train = np.load(SAVE_SEQUENCES / "X_train.npy")
    y_train = np.load(SAVE_SEQUENCES / "y_train.npy")
    print(f"Loaded — X_train: {X_train.shape}")
    print(f"Loaded — y_train: {y_train.shape}")


# Method to transform our X_train in 3-dimensional array of sequences, and y_train in 2D array of corresponding labels.


# ── Core: single sequence ──────────────────────────────────────────────────
def get_Xi_yi(
    fold: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    start_idx: int,
    input_length: int = INPUT_LENGTH,
    horizon: int = HORIZON,
    output_length: int = OUTPUT_LENGTH,
) -> tuple:
    """
    Extract one (X_i, y_i) pair from fold starting at start_idx.

        [start_idx → +input_length)           → X_i  (input_length, n_features)
        [+input_length → +horizon)             → skipped
        [+input_length+horizon → +output_length) → y_i  (output_length,)
    """
    X_i = fold[feature_cols].iloc[start_idx : start_idx + input_length].values
    y_start = start_idx + input_length + horizon
    y_i = fold[target_col].iloc[y_start : y_start + output_length].values
    return X_i, y_i


# parallelized version to extract all sequences at once using numpy's sliding_window_view.
def get_X_y(
    fold: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    stride: int,
    input_length: int = INPUT_LENGTH,
    horizon: int = HORIZON,
    output_length: int = OUTPUT_LENGTH,
) -> tuple:
    """
    Fully vectorized sequence builder using sliding_window_view.
    No Python for loop — all windows created at once.
    """
    total_span = input_length + horizon + output_length
    if len(fold) < total_span:
        raise ValueError(
            f"Fold too short: {len(fold)} rows, need at least {total_span}."
        )

    X_all = fold[feature_cols].values  # (n_rows, n_features)
    y_all = fold[target_col].values  # (n_rows,)

    # all possible windows of size total_span, then subsample by stride
    # shape → (n_windows, total_span, n_features)
    X_wins = np.lib.stride_tricks.sliding_window_view(
        X_all, window_shape=(total_span, X_all.shape[1])
    )[
        :, 0, :, :
    ]  # squeeze extra dim → (n_windows, total_span, n_features)

    # subsample by stride
    X_wins = X_wins[::stride]  # (n_sequences, total_span, n_features)

    # slice X and y out of each window
    X = X_wins[:, :input_length, :]  # (n_seq, input_length, n_features)
    y = X_wins[:, input_length + horizon :, :]  # temp, need target only

    # y from target column directly
    y_wins = np.lib.stride_tricks.sliding_window_view(y_all, total_span)[::stride]
    y = y_wins[:, input_length + horizon :]  # (n_seq, output_length)

    print(
        f"  → {len(X)} sequences  "
        f"(fold={len(fold)}h, stride={stride}h, span={total_span}h)"
    )

    return X, y


if resample_sequences:
    X_train, y_train = get_X_y(
        fold_train,
        feature_cols,
        TARGET_COL,
        STRIDE_TRAIN,
        INPUT_LENGTH,
        HORIZON,
        OUTPUT_LENGTH,
    )
    np.save(SAVE_SEQUENCES / "X_train.npy", X_train)
    np.save(SAVE_SEQUENCES / "y_train.npy", y_train)
    print(f"Sampled and saved — X_train: {X_train.shape}")
    print(f"Sampled and saved — y_train: {y_train.shape}")


# -------------- Validation split from train sequences --------------

print(f"Loaded — X_train: {X_train.shape}")
print(f"Loaded — y_train: {y_train.shape}")

val_split = int(len(X_train) * (1 - VAL_RATIO))

X_tr = X_train[:val_split]
y_tr = y_train[:val_split]
X_val = X_train[val_split:]
y_val = y_train[val_split:]

print(f"\nX_tr:  {X_tr.shape}   y_tr:  {y_tr.shape}")
print(f"X_val: {X_val.shape}  y_val: {y_val.shape}")


# ─────────────────────────────────────────────────────────────────
## LSTM MODEL
# ─────────────────────────────────────────────────────────────────


def initialize_model_lstm(input_shape, output_length):
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(
                128, activation="tanh", return_sequences=True, recurrent_dropout=0.1
            ),  # dropout on recurrent connections
            Dropout(0.2),
            LSTM(64, activation="tanh", return_sequences=True, recurrent_dropout=0.1),
            LSTM(32, activation="tanh", return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dense(output_length, activation="linear"),
        ]
    )
    optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=Huber(delta=1.0),  # behaves like MAE for large errors, MSE for small ones
        metrics=["mae", "mse"],
    )
    model.summary()
    return model


early_stopping = EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True
)


def train_or_load_model_lstm(
    train_new_model,
    model,          # ← already initialized model passed in
    X_tr,
    y_tr,
    X_val,
    y_val,
    EPOCHS,
    BATCH_SIZE,
    MODEL_PATH,
    PATIENCE,
):
    if train_new_model:
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )
        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=1,
        )
        model.save(MODEL_PATH)
        print(f"Trained and saved model to {MODEL_PATH}")
    else:
        print(f"Skipping training, using existing model at {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        history = None

    model.summary()
    return model, history

model_lstm = initialize_model_lstm(input_shape=(INPUT_LENGTH, len(feature_cols)), output_length=OUTPUT_LENGTH)

model_lstm, history_lstm = train_or_load_model_lstm(
    train_new_model,
    model_lstm,
    X_tr,
    y_tr,
    X_val,
    y_val,
    EPOCHS,
    BATCH_SIZE,
    MODEL_PATH,
    PATIENCE,
)


# ---- Create X_test and y_test from fold_test (chronological scan) ----
X_test, y_test = get_X_y(
    fold_test,
    feature_cols,
    TARGET_COL,
    STRIDE_TEST,
    INPUT_LENGTH,
    HORIZON,
    OUTPUT_LENGTH,
)

# Evaluate model on test set
loss, mae, mse = model_lstm.evaluate(X_test, y_test, verbose=1)

print("Test Loss:", loss)
print("Test MAE:", mae)
print("Test MSE:", mse)

target_idx = feature_cols.index(TARGET_COL)

# inverse transform both
y_test_real = y_test * scaler.scale_[target_idx] + scaler.mean_[target_idx]
y_pred = model_lstm.predict(X_test)
y_pred_real = y_pred * scaler.scale_[target_idx] + scaler.mean_[target_idx]

# De-normalized metrics
mae_real = mean_absolute_error(y_test_real, y_pred_real)
mse_real = mean_squared_error(y_test_real, y_pred_real)
rmse_real = np.sqrt(mse_real)

print(f"MAE  (real scale): {mae_real:.2f} ")
print(f"MSE  (real scale): {mse_real:.2f} ")
print(f"RMSE (real scale): {rmse_real:.2f} ")

# print sequence by sequence, hour by hour
# reconstruct start indices the same way get_X_y does
total_span = INPUT_LENGTH + HORIZON + OUTPUT_LENGTH
max_start = len(fold_test) - total_span
starts = list(range(0, max_start + 1, STRIDE_TEST))

# print sequence by sequence, hour by hour
for seq_idx in range(len(y_test_real) - 3, len(y_test_real)):
    start_idx = starts[seq_idx]
    y_start = start_idx + INPUT_LENGTH + HORIZON

    print(f"\n── Sequence {seq_idx} ──────────────────────────────")
    for hour in range(OUTPUT_LENGTH):
        timestamp = fold_test.index[y_start + hour]
        real = y_test_real[seq_idx, hour]
        predicted = y_pred_real[seq_idx, hour]
        print(
            f"  {timestamp} | Real: {real:>8.2f} EUR/MWh | Predicted: {predicted:>8.2f} EUR/MWh"
        )


