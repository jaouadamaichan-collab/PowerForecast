import pandas as pd

def train_test_split_ts(series, split_ratio: float = 0.02):
    """
    Split a time series into training and testing sets while preserving chronological order.

    Parameters
    ----------
    series : pandas.Series
        Time series data indexed by datetime.
    split_ratio : float
        Proportion of data to use as test set (default: 0.2).

    Returns
    -------
    train : pandas.Series
        Training subset (earlier observations).
    test : pandas.Series
        Testing subset (later observations).
    """

    split_index = int(len(series) * (1 - split_ratio))

    train = series[:split_index]
    test = series[split_index:]

    return train, test
