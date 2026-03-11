import pandas as pd
import os
from power_forecast.params import *

# AUTOMATICALLY SAVED IN YOUR LOCAL "raw_data/pickle_files" FOLDER
def save_df(df, name):
    """
    Save a DataFrame as a pickle file in the PICKLE_DIR folder.

    Parameters
    ----------
    df   : DataFrame to save
    name : filename without extension (e.g. 'df_enriched' → 'df_enriched.pkl')
    """
    path = f"{PICKLE_DIR}/{name}.pkl"
    os.makedirs(PICKLE_DIR, exist_ok=True)
    df.to_pickle(path)
    size = os.path.getsize(path)
    print(f"Saved → {path} | {size / 1024 / 1024:.2f} MB")


def load_df(name):
    """
    Load a DataFrame from a pickle file in the PICKLE_DIR folder.

    Parameters
    ----------
    name : filename without extension (e.g. 'df_enriched' → 'df_enriched.pkl')

    Returns
    -------
    df : loaded DataFrame with original index and dtypes preserved
    """
    path = f"{PICKLE_DIR}/{name}.pkl"
    df = pd.read_pickle(path)
    print(f"Loaded ← {path} | {df.shape[0]} rows x {df.shape[1]} columns")
    return df