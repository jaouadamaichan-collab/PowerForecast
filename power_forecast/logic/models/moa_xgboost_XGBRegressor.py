"""
MOA - XGBRegressor Electricity Price Forecast

Objectif
--------
Prédire le prix de l'électricité FRA à horizon +48h.

Dataset
-------
Features déjà préparées dans le dataframe :
- prix européens
- variables temporelles
- météo
- lag features
- rolling statistics
- features avancées de séries temporelles

Deux approches testées :
1. XGBRegressor avec scaling
2. XGBRegressor sans scaling
"""

# resultats obtenus:
#                         baseline_rmse	baseline_mae	train_rmse	train_mae	test_rmse	test_mae
# XGBRegressor_with_scaling	    13.9897	    9.770226	4.137899	2.608593	9.210466	6.264966
# XGBRegressor_without_scaling	13.9897	    9.770226	4.137899	4.353509	9.210466	5.681560
# Remarque : de légères différences entre les résultats du notebook et ceux du .py peuvent apparaître.
# Dans le notebook, les hyperparamètres ont été trouvés via GridSearchCV lors des expérimentations.
# Les paramètres utilisés ici correspondent aux meilleurs résultats trouvés, mais de petites variations
# peuvent subsister selon l'ordre d'exécution du notebook, la préparation des données ou l'état du split train/test.

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor


# =========================================================
# PREPARATION DATASET (split commun)
# =========================================================

def prepare_train_test(df):

    y = df["FRA"]
    X = df.drop(columns=["FRA"])

    return train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )


# =========================================================
# APPROCHE 1 : XGBRegressor AVEC SCALING
# =========================================================

def run_xgb_with_scaling(df):

    X_train, X_test, y_train, y_test = prepare_train_test(df)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(    random_state=42,
    learning_rate=0.05,
    max_depth=4,
    n_estimators=200,
    subsample=0.8
    )

    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    return {
        "approach": "XGBRegressor_with_scaling",
        "model": model,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae
    }


# =========================================================
# APPROCHE 2 : XGBRegressor SANS SCALING
# =========================================================

def run_xgb_without_scaling(df):

    X_train, X_test, y_train, y_test = prepare_train_test(df)

    model = XGBRegressor(    random_state=42,
    learning_rate=0.05,
    max_depth=4,
    n_estimators=200,
    subsample=0.8
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    return {
        "approach": "XGBRegressor_without_scaling",
        "model": model,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae
    }