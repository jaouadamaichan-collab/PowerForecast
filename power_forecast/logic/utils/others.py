# import pandas as pd
# import os
# from power_forecast.params import *

# def save_df(df, name):
#     path = Path(name)

#     # Only prepend PICKLE_DIR if it's a bare filename (no directory)
#     if path.parent == Path("."):
#         path = Path(PICKLE_DIR) / f"{name}.pkl"
#     # else: use the full path as-is, just ensure no double .pkl
#     elif not path.suffix == ".pkl":
#         path = path.with_suffix(".pkl")

#     path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_pickle(path)
#     size = path.stat().st_size
#     print(f"Saved → {path} | {size / 1024 / 1024:.2f} MB")


# def load_df(name):
#     """
#     Load a DataFrame from a pickle file in the PICKLE_DIR folder.

#     Parameters
#     ----------
#     name : filename without extension (e.g. 'df_enriched' → 'df_enriched.pkl')

#     Returns
#     -------
#     df : loaded DataFrame with original index and dtypes preserved
#     """
#     path = f"{PICKLE_DIR}/{name}.pkl"
#     df = pd.read_pickle(path)
#     print(f"Loaded ← {path} | {df.shape[0]} rows x {df.shape[1]} columns")
#     return df
