import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, X_train, y_train, X_test, y_test,
                   X_val=None, y_val=None):
    """
    Calcule les métriques train/val/test et retourne un dict
    compatible avec save_run.

    Parameters
    ----------
    model : objet sklearn/xgboost/keras
        Modèle déjà entraîné avec une méthode predict().
    X_train, y_train : array-like
        Données d'entraînement.
    X_test, y_test : array-like
        Données de test.
    X_val, y_val : array-like, optional
        Données de validation. Optionnel (RNN, LSTM...).

    Returns
    -------
    dict
        {"model": model, "metrics": {...}}
        Directement compatible avec save_run().

    Examples
    --------
    >>> results = evaluate_model(model, X_train, y_train, X_test, y_test)
    >>> results = evaluate_model(model, X_train, y_train, X_test, y_test,
    ...                          X_val=X_val, y_val=y_val)
    """

    metrics = {
        "train_mae" : mean_absolute_error(y_train, model.predict(X_train)),
        "train_rmse": np.sqrt(mean_squared_error(y_train, model.predict(X_train))),
        "test_mae"  : mean_absolute_error(y_test,  model.predict(X_test)),
        "test_rmse" : np.sqrt(mean_squared_error(y_test,  model.predict(X_test))),
    }

    # val optionnel
    if X_val is not None and y_val is not None:
        metrics["val_mae"]  = mean_absolute_error(y_val, model.predict(X_val))
        metrics["val_rmse"] = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

    return {
        "model"  : model,
        "metrics": metrics,
    }