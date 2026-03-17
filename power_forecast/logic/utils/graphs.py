from logging import log
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History
from power_forecast.params import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

def step_label(step: str) -> str:
    return "journalier" if step == "D" else "horaire"


def plot_forecast_sarimax(forecast, X_train, X_test, confidence_int):
    plt.figure(figsize=(12,6))
    plt.plot(X_train, label="train")
    plt.plot(X_test, label="test")
    plt.plot(forecast, label="forecast")

    plt.fill_between(
    confidence_int.index,
    confidence_int.iloc[:,0],
    confidence_int.iloc[:,1],
    alpha=0.7)

    plt.legend()
    plt.show()


def plot_history(history: History):
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


def plot_history_loss_is_mae(history: History):
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
    ax[1].plot(history.history['mse'])
    ax[1].plot(history.history['val_mse'])
    ax[1].set_title('MAE')
    ax[1].set_ylabel('MAE')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='best')
    ax[1].grid(axis="x", linewidth=0.5)
    ax[1].grid(axis="y", linewidth=0.5)

    return ax


def plot_prices(df: pd.DataFrame, step: str, output_dir: Path | None = None) -> None:
    """Génère le graphique des prix (multi-pays si nécessaire)."""
    n_countries = len(df.columns)
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, country in enumerate(df.columns):
        color = PALETTE[i % len(PALETTE)]
        label = COUNTRY_LABELS.get(country, country)
        ax.plot(df.index, df[country], linewidth=1.5, label=label, color=color, alpha=0.85)

    # Mise en forme
    ax.set_title(
        f"Prix day-ahead — {', '.join(COUNTRY_LABELS.get(c, c) for c in df.columns)}\n"
        f"({df.index[0].strftime('%d/%m/%Y')} → {df.index[-1].strftime('%d/%m/%Y')}"
        f" | pas : {step_label(step)})",
        fontsize=13,
        pad=12,
    )
    ax.set_ylabel("€/MWh", fontsize=11)
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=8))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.85)

    if n_countries == 1:
        # Zone colorée sous la courbe pour un pays unique
        country = df.columns[0]
        ax.fill_between(df.index, df[country], alpha=0.08, color=PALETTE[0])

    plt.tight_layout()

    if output_dir:
        path = output_dir / "prix_day_ahead.png"
        fig.savefig(path, dpi=150)
        log.info("📊 Graphique sauvegardé : %s", path)

    plt.show()


def plot_forecast_xgboost(y_train, y_val, y_test, y_val_pred, y_test_pred):

    # params = [y_train, y_val, y_test, y_val_pred, y_val_test]

    # for param in params:
    #     f'{param}_day' = f'{param}'.resample('D').mean()

    index_val = y_val.index
    index_test = y_test.index

    y_val_pred = pd.Series(y_val_pred, index=index_val)
    y_test_pred = pd.Series(y_test_pred, index=index_test)

    y_train_day = y_train.resample('D').mean()
    y_val_day = y_val.resample('D').mean()
    y_test_day = y_test.resample('D').mean()
    y_val_pred_day = y_val_pred.resample('D').mean()
    y_val_test_day = y_test_pred.resample('D').mean()


    plt.figure(figsize=(12,6))
    plt.plot(y_train_day, label="train")
    plt.plot(y_test_day, label="test")
    plt.plot(y_val_pred_day, label="val_pred")
    plt.plot(y_val_test_day, label="test_pred")


import matplotlib.pyplot as plt
import numpy as np

def plot_predictions_rnn(X_test, y_test_real, y_pred_real, feature_cols, TARGET_COL, n_samples=200):
    target_idx = feature_cols.index(TARGET_COL)

    # Extract target column from X_test (last INPUT_LENGTH steps, take the last one)
    x_target = X_test[:n_samples, -1, target_idx]  # last timestep of input window
    y_true    = y_test_real[:n_samples, 0]          # first horizon step
    y_pred    = y_pred_real[:n_samples, 0]

    x_axis = np.arange(n_samples)

    plt.figure(figsize=(14, 5))
    plt.plot(x_axis, x_target, label="Input (last known)", color="steelblue", linewidth=1.2)
    plt.plot(x_axis, y_true,   label="Ground truth",       color="seagreen",  linewidth=1.5)
    plt.plot(x_axis, y_pred,   label="Prediction",         color="tomato",    linewidth=1.5, linestyle="--")
    plt.title(f"Forecast vs Ground Truth — {TARGET_COL}")
    plt.xlabel("Sample index")
    plt.ylabel(TARGET_COL)
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_plot.png", dpi=150)  # always save
    plt.show()

def plot_best_predictions(y_test_real, y_pred_real, TARGET_COL, n_best=5, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)

    # Pick n_best samples where prediction is closest to ground truth
    errors = np.abs(y_test_real[:, 0] - y_pred_real[:, 0])
    best_indices = np.argsort(errors)[:n_best]

    fig, axes = plt.subplots(n_best, 1, figsize=(14, 3 * n_best))
    fig.suptitle(f"Best Predictions — {TARGET_COL}", fontsize=14)

    for ax, idx in zip(axes, best_indices):
        ax.plot(y_test_real[idx], label="Ground truth", color="seagreen", linewidth=1.5)
        ax.plot(y_pred_real[idx], label="Prediction",   color="tomato",   linewidth=1.5, linestyle="--")
        ax.set_title(f"Sample {idx} — MAE: {errors[idx]:.2f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "best_predictions.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
