from pathlib import Path
import pandas as pd

DATA_DIR = Path('data/')
FIGURE_DIR = Path('figs/')
NHS_DATA_FILE = Path('nhs_hospital_admissions_by_date.csv')
WEATHER_DATA_FILE = Path('wales_weather_2020-2023.json')
MERGED_DATA_FILE = Path('wales_weather_and_hospitalizations.feather')
MODEL_DIR = Path('models/')


def read_data() -> pd.DataFrame:
    return pd.read_feather(DATA_DIR / MERGED_DATA_FILE)

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # only keep humidity, precip, pressure, temp, windgust, windspeed, winddir, solarradiation, solarenergy, uvindex
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
        'solarradiation',
        'solarenergy',
        'uvindex'
    ]
    data = data[features]
    # average the weather data per day to match admissions stats
    data = data.resample('D').mean(numeric_only = True)

    # get min Date that has NaNs
    first_nan_date = data[data.isna().any(axis=1)].index[0]

    # remove rows after first NaN
    data = data.loc[data.index < first_nan_date]

    # add day of week: e.g. Monday, Tuesday, etc. as a categorical variable
    data['day_of_week'] = data.index.dayofweek  # type: ignore

    return data, features
