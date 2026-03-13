import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History

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

