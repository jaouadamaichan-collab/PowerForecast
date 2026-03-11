import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_forecast(forecast, X_train, X_test, confidence_int):
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
