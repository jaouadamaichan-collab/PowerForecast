import numpy as np
import pandas as pd

# A utiliser avec le raw_data all_countries.csv

def create_dataframe_base(filepath: str) -> pd.DataFrame:
    """
    Transforms raw electricity price CSV into a time series DataFrame.
    
    - Index: UTC datetime
    - Columns: one per country (ISO3 code)
    - Values: Price (EUR/MWhe)
    """
    # Load raw CSV
    df = pd.read_csv("/Users/Mohamed.Atrari/code/jaouadamaichan-collab/PowerForecast/raw_data/all_countries.csv")
    
    # Parse UTC datetime & set as index
    df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'], utc=True)
    
    # Pivot: rows = datetime, columns = country, values = price
    df_pivot = df.pivot_table(
        index='Datetime (UTC)',
        columns='ISO3 Code',
        values='Price (EUR/MWhe)',
        fill_value = np.nan, 
        aggfunc='first',
        dropna=False 
    )
    
    # Clean column name metadata
    df_pivot.columns.name = None
    df_pivot.index.name = 'datetime_utc'
    
    # Sort index
    df_pivot = df_pivot.sort_index()
    df_pivot = df_pivot.drop('MKD', axis=1)
    
    return df_pivot


def align_start_to_column(df, column, apply=True):
    """
    Make dataset start from the first valid (non-NaN) value of a specific column.
    If Croatia: 2017-10-01 00:00:00
    """
    if apply:
        first_valid = df[column].first_valid_index()
        df_aligned = df[df.index >= first_valid]
        print(f"New start: {first_valid} | Dropped {len(df) - len(df_aligned)} rows | {len(df_aligned)} remaining")
        return df_aligned
    return df

def replace_outliers_with_interpolation(df, limit_low, limit_high):
    """
    Replace values outside [limit_low, limit_high] with NaN,
    then interpolate all NaNs
    """

    # Replace outliers with NaN
    df_clean = df.copy()
    df_clean[df_clean < limit_low] = float('nan')
    df_clean[df_clean > limit_high] = float('nan')

    # Interpolate ALL NaNs (pre-existing + new outliers)
    df_interpolated = df_clean.interpolate(method='time')

    return df_interpolated


