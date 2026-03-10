import pandas as pd
from sklearn.preprocessing import StandardScaler
from power_forecast.logic.get_data.download_api import replace_outliers_with_interpolation, align_start_to_column

def preproc_arima(df, column):

    #cleaned from ancient nan values
    df_nonan = align_start_to_column(df=df, column=column)

    #cleaned from outliers
    df_clean = replace_outliers_with_interpolation(df, limit_low=-350, limit_high=2000)
    #standard scaling
    scaler = StandardScaler().set_output(transform='pandas')
    df_scaled = scaler.fit_transform(df_clean)

    #extract one country
    df_country = df_scaled['column']

    #resample for the mean of a day
    df_country_day = df_country.resample('D').mean()

    return df_country_day
