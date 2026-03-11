import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def init_fit_model_(X_train, X_test):
    sarimax = SARIMAX(X_train, order=(2, 1, 2), seasonal_order=(1, 0, 1, 7))
    sarimax = sarimax.fit(maxiter=70)

    forecast_results = sarimax.get_forecast(len(X_test))
    forecast = forecast_results.predicted_mean
    confidence_int = forecast_results.conf_int(alpha=0.05)

    return forecast, confidence_int
