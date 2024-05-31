from pathlib import Path
from math import ceil, cos, radians, sin
import pandas as pd
import logging

DATA_DIR = Path('data/')
FIGURE_DIR = Path('figs/')
NHS_DATA_FILE = Path('nhs_hospital_admissions_by_date.csv')
WEATHER_DATA_FILE = Path('wales_weather_2020-2023.json')
MERGED_DATA_FILE = Path('wales_weather_and_hospitalizations.feather')
MODEL_DIR = Path('models/')


def read_data() -> pd.DataFrame:
    return pd.read_feather(DATA_DIR / MERGED_DATA_FILE)

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    features = [
        'COVID-19 admissions (suspected and confirmed)',
        'Non-COVID-19 admissions',
        'Total admissions',
        'humidity',
        'precip',
        'pressure',
        'temp',
        'windgust',
        'windspeed',
        'winddir',
        'snow',
        'snowdepth',
        'solarradiation',
        'solarenergy',
        'uvindex',
    ]
    data = data[features]

    # get min Date that has NaNs
    first_nan_date = data[data.isna().any(axis=1)].index[0]
    
    # remove rows after first NaN
    n_rows = data.shape[0]
    first_date = data.index[0]
    last_date = data.index[-1]
    data = data.loc[data.index < first_nan_date]
    if data.shape[0] < n_rows:
        logging.warn(
            f'Rows with NaNs excluded\nRemoved {n_rows - data.shape[0]}'
            f' rows ({ceil((n_rows - data.shape[0])/24)} days) with NaNs\nOriginal'
            f' date range:\n\t{first_date} to {last_date}\nNew date range:\n\t'
            f'{data.index.min()} to {data.index.max()}'
        )


    # get mean, min, max of weather data by day
    data = data.resample('D').agg({
        'COVID-19 admissions (suspected and confirmed)': 'first', # daily
        'Non-COVID-19 admissions': 'first', # daily
        'Total admissions': 'first', # daily
        'humidity': ['mean', 'min', 'max', 'std'], # hourly
        'precip': ['sum', 'max'], # hourly
        'pressure': ['mean', 'min', 'max', 'std'], # hourly
        'temp': ['mean', 'min', 'max', 'std'], # hourly
        'windgust': ['mean', 'max', 'std'], # hourly
        'windspeed': ['mean', 'max', 'std'], # hourly
        'winddir': 'mean',
        'snow': 'max', # hourly
        'snowdepth': 'max', # hourly
        'solarradiation': 'max', # hourly
        'solarenergy': 'max', # hourly
        'uvindex': 'max', # hourly
    })

    # replace NaN values in the 'std' columns with 0.0
    # there aren't many but it breaks analysis
    std_cols = [col for col in data.columns if 'std' in col]
    data[std_cols] = data[std_cols].fillna(0.0)
    # add day of week: e.g. Monday, Tuesday, etc. as a categorical variable
    data['day_of_week'] = data.index.dayofweek  # type: ignore

    # flatten the multi-index columns
    data.columns = ['_'.join(col).strip() for col in data.columns.values]  # type: ignore

    data.rename(
        columns={
            'COVID-19 admissions (suspected and confirmed)_first': 'cov19',
            'Non-COVID-19 admissions_first': 'noncov19',
            'Total admissions_first': 'total',
            'day_of_week_': 'day_of_week',
        },
        inplace=True
    )

    # add winddir_sin and winddir_cos
    data['winddir_sin'] = data['winddir_mean'].apply(lambda x: sin(radians(x)))
    data['winddir_cos'] = data['winddir_mean'].apply(lambda x: cos(radians(x)))

    return data
