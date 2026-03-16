from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def run_and_log(train_function, *args, **kwargs):

    output = train_function(*args, **kwargs)

    # cas 1 : modèle retourne déjà un dict
    if isinstance(output, dict):
        results = output

    # cas 2 : modèle retourne seulement le modèle
    else:
        results = {"model": output}

    model = results["model"]

    # essayer de calculer les métriques si possible
    if hasattr(model, "predict") and len(args) >= 2:

        X = args[0]
        y = args[1]

        try:
            y_pred = model.predict(X)

            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)

            results["rmse"] = rmse
            results["mae"] = mae

        except Exception:
            pass

    return results