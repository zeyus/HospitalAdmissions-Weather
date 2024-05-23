# https://statswales.gov.wales/Catalogue/Health-and-Social-Care/NHS-Hospital-Activity/nhs-activity-and-capacity-during-the-coronavirus-pandemic/admissions-by-date-patientype
# licence: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
# hospital admissions by wales health board.
# Columns: "Date","COVID-19 admissions (suspected and confirmed)","Non-COVID-19 admissions","Total admissions"
# path data/nhs_hospital_admissions_by_date.csv

# Weather data is from visualcrossing, in json format.
# Path: data/wales_weather_2020-2023.json


## Baseline: ARIMA model
## Comparison: LSTM model, Neural Prophet model?
## Treat as a forecasting problem, perhaps there's
## - Seasonality
## - Lag between weather and hospitalizations
## - 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import tqdm
from pathlib import Path
import logging
from datetime import datetime
from common.config import DATA_DIR, FIGURE_DIR, NHS_DATA_FILE, WEATHER_DATA_FILE, MERGED_DATA_FILE




def prepare_data() -> pd.DataFrame:
    # Load the data
    logging.info('Loading NHS data')
    hospital_data = pd.read_csv(DATA_DIR / NHS_DATA_FILE)
    # date is in the format dd Month yyyy
    
    hospital_data['Date'] = pd.to_datetime(hospital_data['Date'], format='%d %B %Y', utc=False)
    hospital_data.set_index('Date', inplace=True)

    # Load the weather data
    logging.info('Loading weather data')
    with open(DATA_DIR / WEATHER_DATA_FILE, 'r') as f:
        weather = json.load(f)

    daily_data = weather["days"]
    # daily data contains the following that we care about:
    # datetime (YYYY-MM-DD)
    # hours (list of hourly data)
    #  - datetime (HH:MM:SS)
    #  - temp (temperature in celsius)
    #  - precip (precipitation in mm)
    #  - preciptype (type of precipitation)
    #  - humidity (percentage)
    #  - windspeed (wind speed in km/h)
    #  - windgust (wind gust in km/h)
    #  - winddir (wind direction in degrees)
    #  - pressure (pressure in hPa)
    #  - solarradiation (solar radiation in W/m^2)
    #  - solarenergy (solar energy in J/m^2)
    #  - uvindex (UV index)
    #  - stations (list of weather stations)
    #  - source (e.g. "obs")
    #  - snow (snow in mm)
    #  - snowdepth (snow depth in mm)
    #  - dew (dew point in celsius)

    # now we put the above data into a dataframe
    logging.info('Creating weather dataframe')
    weather_df = pd.DataFrame(
        columns=[
            'datetime',
            'temp',
            'precip',
            'preciptype',
            'humidity',
            'windspeed',
            'windgust',
            'winddir',
            'pressure',
            'solarradiation',
            'solarenergy',
            'uvindex',
            'snow',
            'snowdepth',
            'dew',
            'stations',
        ]
    )

    for day in tqdm.tqdm(daily_data, desc='Processing weather data', unit='days', colour='red'):
        for hour in day['hours']:
            weather_df = pd.concat(
                [
                    pd.DataFrame(
                        [[
                            ' '.join([day['datetime'], hour['datetime']]),
                            hour['temp'],
                            hour['precip'],
                            hour['preciptype'],
                            hour['humidity'],
                            hour['windspeed'],
                            hour['windgust'],
                            hour['winddir'],
                            hour['pressure'],
                            hour['solarradiation'],
                            hour['solarenergy'],
                            hour['uvindex'],
                            hour['snow'],
                            hour['snowdepth'],
                            hour['dew'],
                            hour['stations'],
                        ]],
                        columns=weather_df.columns),
                    weather_df
                ],
                ignore_index=True
            )

    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'], format='%Y-%m-%d %H:%M:%S', utc=False)
    weather_df.set_index('datetime', inplace=True)

    # now we merge the two dataframes, daily data will be repeated for each hour
    # we must match on date only, time is not in the hospital data
    logging.info('Sorting dataframes')
    hospital_data.sort_index(inplace=True)
    weather_df.sort_index(inplace=True)

    logging.info('Merging dataframes')
    merged_data = pd.merge_asof(
        hospital_data,
        weather_df,
        left_index=True,
        right_index=True,
        direction='nearest'
    )

    # save the merged data
    logging.info('Saving merged data')
    merged_data.to_feather(DATA_DIR / MERGED_DATA_FILE)

    return merged_data


def plot_data_exploration(hospital_weather: pd.DataFrame):
    # first, let's get daily min, max, and mean temperatures
    daily_temp = hospital_weather['temp'].resample('1D').mean()
    daily_temp_min = hospital_weather['temp'].resample('1D').min()
    daily_temp_max = hospital_weather['temp'].resample('1D').max()

    # let's just look at september 2020 to limit the data
    daily_temp = daily_temp['2020-09']
    daily_temp_min = daily_temp_min['2020-09']
    daily_temp_max = daily_temp_max['2020-09']

    # now get the admissions data, we only need to pick the first value of each day (since it's daily data)
    start_date = datetime(2020, 9, 1)
    end_date = datetime(2020, 9, 30)
    hospital_weather_daily = hospital_weather.resample('1D').first()

    # now only include all the days in september 2020
    hospital_weather_daily = hospital_weather_daily[start_date:end_date]

    
    fig, ax = plt.subplots()
    ax.plot(daily_temp, label='Mean Temp', color='orange', linestyle='dashed')
    ax.plot(daily_temp_min, label='Min Temp', color='blue', linestyle='dashed')
    ax.plot(daily_temp_max, label='Max Temp', color='red', linestyle='dashed')
    ax2 = ax.twinx()
    ax2.plot(hospital_weather_daily['COVID-19 admissions (suspected and confirmed)'], label='COVID-19 Admissions', color='purple')
    ax2.plot(hospital_weather_daily['Non-COVID-19 admissions'], label='Non-COVID-19 Admissions', color='green')
    ax.legend()
    ax2.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (C)')
    ax2.set_ylabel('Admissions')
    ax.set_title('Temperature vs Admissions')
    fig.savefig(FIGURE_DIR / 'temp_vs_admissions.png')

    # now let's look at the correlation between the variables
    # rename admissions to shorter names
    hospital_weather.rename(
        columns={
            'COVID-19 admissions (suspected and confirmed)': 'C-19',
            'Non-COVID-19 admissions': 'Non-C19',
            'Total admissions': 'Total'
        },
        inplace=True
    )
    fig, ax = plt.subplots()
    ax.matshow(hospital_weather.select_dtypes(include=[np.number]).corr())
    ax.set_xticks(range(hospital_weather.shape[1]))
    ax.set_xticklabels(hospital_weather.columns, rotation=60)
    ax.set_yticks(range(hospital_weather.shape[1]))
    ax.set_yticklabels(hospital_weather.columns, rotation=30)
    # add padding to not cut off the tick labels
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'correlation_matrix.png')




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    hospital_weather: pd.DataFrame

    if not (DATA_DIR / MERGED_DATA_FILE).exists():
        logging.info('Merged data does not exist, preparing data')
        hospital_weather = prepare_data()
    else:
        logging.info('Merged data already exists, loading data')
        hospital_weather = pd.read_feather(DATA_DIR / MERGED_DATA_FILE)

    logging.info('Plotting data exploration')
    plot_data_exploration(hospital_weather)
    logging.info('Done')

    