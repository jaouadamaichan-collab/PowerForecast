import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_validate


def init_fit_histxgb(X: pd.DataFrame, y: pd.Series):
    model = HistGradientBoostingRegressor(max_iter=300)
    model_trained = model.fit(X, y)
    # save_model_ml(model_trained, model_name='HistXGB_v2')

    return model_trained

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    #evaluating model on train
    cv_results = cross_validate(model,
                            X_train,
                            y_train,
                            cv=10,
                            scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error'],
                            n_jobs=-1)

    mae_train = round(abs(cv_results['test_neg_mean_absolute_error'].mean()),2)
    rmse_train = round(abs(cv_results['test_neg_root_mean_squared_error'].mean()),2)

    #evaluating model on val
    y_val_pred = model.predict(X_val)
    mae_val = round(mean_absolute_error(y_val, y_val_pred),2)
    rmse_val = round(root_mean_squared_error(y_val, y_val_pred),2)

    #evaluating model on test
    y_test_pred = model.predict(X_test)
    mae_test = round(mean_absolute_error(y_test, y_test_pred),2)
    rmse_test = round(root_mean_squared_error(y_test, y_test_pred),2)

    return pd.DataFrame({
    "Set": ["Train", "Validation", "Test"],
    "MAE": [mae_train, mae_val, mae_test],
    "RMSE": [rmse_train, rmse_val, rmse_test] })
