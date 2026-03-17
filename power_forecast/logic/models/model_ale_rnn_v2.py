from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
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
from power_forecast.logic.get_data.build_dataframe import build_common_dataframe, add_features_XGB, add_features_RNN
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from power_forecast.params import *
from power_forecast.logic.get_data.build_dataframe import (
    build_common_dataframe,
    add_features_RNN,
)
from power_forecast.logic.preprocessing.train_test_split import (
    train_test_split_general,
    train_test_split_RNN_optimized,
)
from power_forecast.logic.preprocessing.split_X_y_standardize import (
    get_X_y_vectorized_RNN,
    get_Xi_yi_single_sequence,
)
from power_forecast.logic.utils.graphs import plot_predictions_rnn, plot_best_predictions
#pd.set_option("display.max_columns", None)


## PARAMÈTRES DE STRATÉGIE D'ENTRAÎNEMENT
max_train_test_split = True

objective_day = pd.Timestamp("2024-03-24", tz="UTC")

cutoff_day = pd.Timestamp("2023-10-01", tz="UTC")


# Other inputs
input_length = 14 * 24  # 3 weeks context fed to RNN
stride_sequences = 24 * 3  # doit etre plus haute que output length
prediction_horizon_days = 2
country_price_objective = "France"
prediction_length = prediction_horizon_days * 24  # predict 48h of target day

#Model hyperparam
train_new_model = True
patience_model = 10
batch_size_model = 32
epochs_model = 100
version = "2"

MODEL_DIR = Path("raw_data/pickle_files/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_name = f"{country_price_objective}_batch{batch_size_model}_input{input_length}_v_{version}"
model_path  = MODEL_DIR / f"{model_name}.keras"




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


df_common = build_common_dataframe(
    filepath="raw_data/all_countries.csv",
    country_objective=country_price_objective,
    target_day_distance=prediction_horizon_days,
    time_interval="h",
    keep_only_neighbors=True,
    add_meteo=True,
    add_crisis=True,
    add_entsoe=True,
)

df = add_features_RNN(
    df=df_common,
    country_objective=country_price_objective,
    target_day_distance=prediction_horizon_days,
    add_future_time_features=True,
    add_future_meteo=True,
)

columns_rnn = df.columns
print(f"      Shape of all data{df.shape}")


# if max_train_test_split = True il train jusqu'a derniere moment possible basè sur objective_day
if max_train_test_split:
    # RNN
    fold_train_rnn, fold_test_rnn = train_test_split_RNN_optimized(
        df=df,
        objective_day=objective_day,
        number_days_to_predict=prediction_horizon_days,
        input_length=input_length,  # 168h lookback
    )
if not max_train_test_split:
    # RNN
    fold_train_rnn, fold_test_rnn = train_test_split_general(df=df, cutoff=cutoff_day)


scaler = StandardScaler()

# ── Train ──────────────────────────────────────────────────────────────────
X_train, y_train = get_X_y_vectorized_RNN(
    fold=fold_train_rnn,
    feature_cols=fold_train_rnn.columns,
    country_objective=country_price_objective,
    stride=stride_sequences,
    input_length=input_length,
    output_length=prediction_length,
    scaler=scaler,
    fit_scaler=True,
)

# ── Test ───────────────────────────────────────────────────────────────────
X_new, y_true = get_X_y_vectorized_RNN(
    fold=fold_test_rnn,
    feature_cols=fold_test_rnn.columns,
    country_objective=country_price_objective,
    stride=stride_sequences,
    input_length=input_length,
    output_length=prediction_length,
    scaler=scaler,
    fit_scaler=False,
)

# # ── X_new : dernière séquence du fold_test pour prédiction ────────────────
# X_new = X_test[-1:]  # (1, input_length, n_features) -> deja bon dimension

if max_train_test_split:
    print("📐 Shapes finales :")
    print(f"    X_train: {X_train.shape} → (n_seq, input_length, n_features)")
    print(f"    y_train: {y_train.shape} → (n_seq, output_length)")
    print(f"    X_new: {X_new.shape} → (1, input_length, n_features)")
    print(f"    y_true: {y_true.shape}→ (n_seq, output_length)")

# ── Validation : split chronologique SUR LES SÉQUENCES (pas sur le fold brut)
if not max_train_test_split:
    val_ratio = 0.2
    split_idx = int(len(X_train) * (1 - val_ratio))

    X_val = X_train[split_idx:]  # séquences val → suivent chronologiquement train
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]  # on réduit X_train en conséquence
    y_train = y_train[:split_idx]
    print("📐 Shapes finales :")
    print(f"    X_train: {X_train.shape} → (n_seq, input_length, n_features)")
    print(f"    y_train: {y_train.shape} → (n_seq, output_length)")
    print(f"    X_val: {X_val.shape} → (n_seq, input_length, n_features)")
    print(f"    y_val: {y_val.shape} → (n_seq, output_length)")
    print(f"    X_test: {X_new.shape} → (1, input_length, n_features)")
    print(f"    y_test: {y_true.shape}→ (n_seq, output_length)")

input_shape=X_train.shape[1:]
output_length=y_train.shape[1]

model_lstm = initialize_model_lstm(input_shape=input_shape, output_length=output_length)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=patience_model, restore_best_weights=True
)

force_retrain = False  # ← passe à True pour ignorer le cache
if max_train_test_split:
    if model_path.exists() and not force_retrain:
        print(f"✅ Modèle existant chargé : {model_path}")
        model_lstm = load_model(model_path)

    else:
        print(f"🏋️ Aucun modèle trouvé — entraînement en cours...")
        history = model_lstm.fit(
            X_train,
            y_train,
            epochs=epochs_model,
            batch_size=batch_size_model,
            callbacks=[early_stopping],
            verbose=1,
        )
        model_lstm.save(model_path)
        print(f"✅ Modèle sauvegardé : {model_path}")
        
if not max_train_test_split:
    if model_path.exists() and not force_retrain:
        print(f"✅ Modèle existant chargé : {model_path}")
        model_lstm = load_model(model_path)

    else:
        print(f"🏋️ Aucun modèle trouvé — entraînement avec validation en cours...")
        history = model_lstm.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_model,
            batch_size=batch_size_model,
            callbacks=[early_stopping],
            verbose=1,
        )
        model_lstm.save(model_path)
        print(f"✅ Modèle sauvegardé : {model_path}")


if max_train_test_split:
    y_pred_rnn  = model_lstm.predict(X_new, verbose=0).flatten()
    y_true_flat = y_true.flatten()

    time_index  = pd.date_range(
        start   = objective_day,
        periods = output_length,
        freq    = "h",
        tz      = "UTC"
    )
    y_true_plot = y_true_flat
    y_pred_plot = y_pred_rnn

if not max_train_test_split:
    print("📊 Évaluation sur le jeu de test :")
    results = model_lstm.evaluate(X_test, y_true, verbose=1)
    print(f"   Huber Loss : {results[0]:.4f}")
    print(f"   MAE        : {results[1]:.4f}")
    print(f"   MSE        : {results[2]:.4f}")

    y_pred_rnn  = model_lstm.predict(X_test, verbose=0).flatten()
    y_true_flat = y_true.flatten()

    time_index  = pd.date_range(
        start   = fold_test.index[input_length],
        periods = output_length,
        freq    = "h",
        tz      = "UTC"
    )
    y_true_plot = y_true[0]                    # première séquence uniquement
    y_pred_plot = y_pred_rnn[:output_length]   # idem

# ── Métriques (toutes séquences) ──────────────────────────────────────────
mae  = mean_absolute_error(y_true_flat, y_pred_rnn)
mse  = mean_squared_error(y_true_flat, y_pred_rnn)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true_flat - y_pred_rnn) / (y_true_flat + 1e-8))) * 100

print("\n📐 Métriques finales :")
print(f"   MAE  : {mae:.4f}")
print(f"   MSE  : {mse:.4f}")
print(f"   RMSE : {rmse:.4f}")
print(f"   MAPE : {mape:.2f} %")

# ── Plot (première séquence dans else, objective_day dans if) ─────────────
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(time_index, y_true_plot, label="🔵 y_true",     color="#4C9BE8", linewidth=2)
ax.plot(time_index, y_pred_plot, label="🟠 y_pred RNN", color="#F4845F", linewidth=2, linestyle="--")
ax.fill_between(time_index, y_true_plot, y_pred_plot, alpha=0.08, color="#A78BFA")

ax.set_title(
    f"Prédiction RNN — {country_price_objective}  |  "
    f"MAE {mae:.2f}  •  RMSE {rmse:.2f}  •  MAPE {mape:.1f}%",
    fontsize=13, fontweight="bold", pad=14
)
ax.set_xlabel("Temps (UTC)", fontsize=11)
ax.set_ylabel("Prix (€/MWh)", fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()