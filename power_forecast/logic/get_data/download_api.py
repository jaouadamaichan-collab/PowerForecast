import numpy as np
import pandas as pd

import pycatch22


# A utiliser avec le raw_data all_countries.csv
def create_dataframe_base(filepath: str) -> pd.DataFrame:
    """
    Transforms raw electricity price CSV into a time series DataFrame.

    - Index: UTC datetime
    - Columns: one per country (ISO3 code)
    - Values: Price (EUR/MWhe)
    """
    # Load raw CSV

    df = pd.read_csv(filepath)
    # Parse UTC datetime & set as index
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"], utc=True)

    # Pivot: rows = datetime, columns = country, values = price
    df_pivot = df.pivot_table(
        index="Datetime (UTC)",
        columns="ISO3 Code",
        values="Price (EUR/MWhe)",
        fill_value=np.nan,
        aggfunc="first",
        dropna=False,
    )

    # Clean column name metadata
    df_pivot.columns.name = None
    df_pivot.index.name = "datetime_utc"

    # Sort index
    df_pivot = df_pivot.sort_index()
    df_pivot = df_pivot.drop("MKD", axis=1)

    return df_pivot

