import pandas as pd
from power_forecast.params import *


def prepare_X_new(date_str: str, test_fold: pd.DataFrame, horizon: int = HORIZON, input_length: int = INPUT_LENGTH) -> pd.DataFrame:
    """
    Prepare X_new for predicting electricity prices for a given target day.

    Given a target date, this function extracts the input window from test_fold
    that the model needs to make predictions for all 24 hours of that day.

    Parameters
    ----------
    date_str     : str
        Target date in 'DD-MM-YYYY' format (e.g., '20-03-2024').
    test_fold    : pd.DataFrame
        The test DataFrame containing features. Must have a DatetimeIndex.
        ⚠️  WARNING: test_fold must NEVER contain any rows from train_fold.
        Make sure the split is done before calling this function, otherwise
        you will leak training data into your prediction input.
    horizon      : int
        Number of hours before the target day where the input window ends.
        e.g., horizon=24 means the last timestamp of X_new is 23:00 two days before.
    input_length : int
        Number of hours to look back from the end of the input window.
        e.g., input_length=24*14 means 2 weeks of hourly data as input.

    Returns
    -------
    X_new : pd.DataFrame
        Input window of shape (input_length, n_features), with DatetimeIndex
        ranging from (target_day - horizon hours - input_length hours)
        to (target_day - horizon hours - 1 hour).

    Example
    -------
    For target_date = 20 March, horizon=24, input_length=24*14:
        - Last timestamp  of X_new : 23:00 on 18 March
        - First timestamp of X_new : 00:00 on  5 March
    """
    # Parse target date
    target_date = pd.to_datetime(date_str, format='%d-%m-%Y')

    # End of input window: (target_day - horizon) -> last hour before horizon
    # e.g., target = 20 March 00:00, horizon=24 -> end = 18 March 23:00
    end_dt = target_date - pd.Timedelta(hours=horizon) + pd.Timedelta(hours=23)

    # Start of input window
    start_dt = end_dt - pd.Timedelta(hours=input_length - 1)

    print(f"  Target date       : {target_date.date()}")
    print(f"  X_new start       : {start_dt}")
    print(f"  X_new end         : {end_dt}")
    print(f"  Window length     : {input_length}h ({input_length // 24} days)")

    # Check the window is fully covered by test_fold
    if start_dt < test_fold.index.min():
        raise ValueError(
            f"start_dt {start_dt} is before test_fold start {test_fold.index.min()}. "
            f"Either reduce input_length or check your train/test split."
        )

    X_new = test_fold.loc[start_dt:end_dt]

    if len(X_new) != input_length:
        raise ValueError(
            f"Expected {input_length} rows but got {len(X_new)}. "
            f"There may be missing timestamps in test_fold between {start_dt} and {end_dt}."
        )

    return X_new