import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from power_forecast.params import *


def plot_forecast_xgboost(y_train, y_val, y_test, y_val_pred, y_test_pred):
    index_val = y_val.index
    index_test = y_test.index

    y_val_pred = pd.Series(y_val_pred, index=index_val)
    y_test_pred = pd.Series(y_test_pred, index=index_test)

    y_train_day = y_train.resample('D').mean()
    y_true_day = pd.concat([y_val, y_test]).resample('D').mean()
    y_pred_day = pd.concat([y_val_pred, y_test_pred]).resample('D').mean()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(y_train_day, label="Train", color="#4C72B0", linewidth=1.5, alpha=0.85)
    ax.plot(y_true_day, label="True", color="#55A868", linewidth=1.5, alpha=0.9)
    ax.plot(y_pred_day, label="Predicted", color="#C44E52", linewidth=1.5, linestyle="--", alpha=0.9)

    # Séparateur train / prévision
    split_date = y_true_day.index[0]
    ax.axvline(split_date, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(split_date, ax.get_ylim()[1], " début prévision", fontsize=8, color="gray", va="top")

    ax.set_title("Prévision XGBoost — valeurs journalières moyennes", fontsize=13, pad=12)
    ax.set_ylabel("€/MWh", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=8))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    plt.show()
