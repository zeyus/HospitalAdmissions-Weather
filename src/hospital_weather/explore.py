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
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.ticker import FuncFormatter, IndexLocator, LinearLocator
import matplotlib.pyplot as plt
import numpy as np
import json
import tqdm
import logging
import random
import seaborn as sns
from json import dump
from .common.config import prepare_data, DATA_DIR, FIGURE_DIR, NHS_DATA_FILE, WEATHER_DATA_FILE, MERGED_DATA_FILE




def load_data() -> pd.DataFrame:
    # Load the data
    logging.info('Loading NHS data')
    hospital_data = pd.read_csv(DATA_DIR / NHS_DATA_FILE)
    # repeat the data for each hour
    hospital_data = hospital_data.reindex(hospital_data.index.repeat(24))
    hospital_data.reset_index(drop=True, inplace=True)

    # now add the hour sequence
    hospital_data['Date'] = hospital_data['Date'] + hospital_data.groupby('Date').cumcount().apply(lambda x: f' {x}:00:00')
    logging.info(hospital_data.head())
    # date is in the format dd Month yyyy
    hospital_data['Date'] = pd.to_datetime(hospital_data['Date'], format='%d %B %Y %H:%M:%S', utc=False, cache=False)
    hospital_data['Date'].dt.to_period('H')
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
    columns=[
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
    data: dict[str, list] = {
        col: [] for col in columns
    }
    index = []
    for day in tqdm.tqdm(daily_data, desc='Processing weather data', unit='days', colour='red'):
        for hour in day['hours']:
            index.append(pd.to_datetime(f"{day['datetime']} {hour['datetime']}", format='%Y-%m-%d %H:%M:%S', utc=False))
            for col in columns:
                data[col].append(hour[col])

    weather_df = pd.DataFrame(data, columns=columns, index=index)

    # now we merge the two dataframes, daily data will be repeated for each hour
    # we must match on date only, time is not in the hospital data
    logging.info('Sorting dataframes')
    hospital_data.sort_index(inplace=True)
    weather_df.sort_index(inplace=True)

    logging.info('Merging dataframes')
    merged_data = pd.merge_asof(
        weather_df,
        hospital_data,
        left_index=True,
        right_index=True,
        direction='nearest',
        allow_exact_matches=True,
        tolerance=pd.Timedelta('1H')
    )

    # save the merged data
    logging.info('Saving merged data')
    merged_data.to_feather(DATA_DIR / MERGED_DATA_FILE)

    return merged_data


def plot_data_exploration(hospital_weather: pd.DataFrame):
    # now let's look at the correlation between the variables
    
    # drop solarradiation_max and uvindex_max
    d = hospital_weather.drop(columns=['day_of_week', 'solarradiation_max', 'precip_max', 'snow_max', 'solarenergy_max', 'total', 'noncov19'], inplace=False)
    corr = d.corr()
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 20)  # type: ignore
    cbar_ax = fig.add_axes((.92, .3, .02, .4))
    heatmap = sns.heatmap(corr, cmap="vlag", ax=ax, square=True, annot=False, vmin=-1.0, vmax=1.0, cbar_ax=cbar_ax)
    
    heatmap.set_title('Variable Correlation Heatmap', fontdict={'fontsize': 30}, pad=12)
    # set x and y rotation
    heatmap.set_xticks(np.arange(len(corr.columns))+0.5, corr.columns, rotation=60, ha='right')
    heatmap.set_yticks(np.arange(len(corr.columns))+0.5, corr.columns, rotation=330, va='bottom')
    plt.savefig(FIGURE_DIR / 'correlation_heatmap.png', bbox_inches='tight', dpi=300)
    # plt.rcParams.update({'font.size': 20})
    # fig, ax = plt.subplots(figsize=(20, 20))
    # cax = ax.matshow(corr, cmap='coolwarm')
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation=300, ha='right')  # type: ignore
    # plt.yticks(range(len(corr.columns)), corr.columns, rotation=330, va='bottom')  # type: ignore
    # plt.legend()
    # fig.colorbar(cax)
    # fig.savefig(FIGURE_DIR / 'correlation_matrix.png', bbox_inches='tight')

    # plot 3 x 3 grid of 6 random months
    fig, axs = plt.subplots(3, 3, figsize=(20, 20), sharey='row')
    ax2s: np.ndarray = np.empty((3, 3), dtype=Axes)
    for i in range(3):
        for j in range(3):
            ax2s[i, j] = axs[i, j].twinx()

    min_date = hospital_weather.index.min()
    max_date = hospital_weather.index.max()
    date_range: pd.DatetimeIndex = pd.date_range(min_date, max_date - pd.DateOffset(months=1), freq='ME')
    handles: list[Artist] = []
    handles2: list[Artist] = []
    labels: list = []
    labels2: list = []
    for i in range(3):
        for j in range(3):
            a: Axes = axs[i, j]
            a2: Axes = ax2s[i, j]
            start: pd.Timestamp = random.choice(date_range)
            # remove selected date from the list
            date_range = date_range[date_range != start]
            end: pd.Timestamp = start + pd.DateOffset(months=1)
            data = hospital_weather.loc[start:end]  # type: ignore
            a.plot(data['cov19'], label='COVID-19 admissions', color='grey')
            # add precipitation
            a.plot(data['precip_sum'], label='Precipitation (total)')
            # add windgust
            a.plot(data['windgust_mean'], label='Wind Gust (mean)')
            # add temperature max
            a.plot(data['temp_max'], label='Temperature (max)')
            # add non-covid admissions on a secondary y-axis
            a2.plot(data['noncov19'], label='Non-COVID-19 admissions', linestyle='--', color='grey')  # type: ignore
            a.xaxis.set_major_locator(IndexLocator(7, 0))
            if i == 2:
                # make x-ticks the day of the month only, not the full date
                a.xaxis.set_major_formatter(FuncFormatter(lambda _, pos: f'{1 + (pos * 7)}'))
                if j == 1:
                    a.set_xlabel('Date')
            else:
                # hide x-axis labels
                a.set_xticklabels([])
                a2.set_xticklabels([])
            
            if j != 2:
                a2.set_yticklabels([])

            if j == 2:
                a2.get_shared_y_axes().joined(a2, ax2s[i, 0])
                a2.get_shared_y_axes().joined(a2, ax2s[i, 1])
                if i == 1:
                    a2.set_ylabel('Non-COVID-19 Admissions')
            elif j == 0 and i == 1:
                a.set_ylabel('Values')
                


            a.set_title(f'{start.strftime("%B %Y")} to {end.strftime("%B %Y")}')

            if i == 0 and j == 0:
                handles, labels = a.get_legend_handles_labels()
                # add the twin axis labels
                handles2, labels2 = a2.get_legend_handles_labels()  # type: ignore


    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.05, 0.97))
    fig.legend(handles2, labels2, loc='upper right', bbox_to_anchor=(0.95, 0.97))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'admissions_by_month_sample.png')



def explore_data() -> None:
    hospital_weather: pd.DataFrame

    if not (DATA_DIR / MERGED_DATA_FILE).exists():
        logging.info('Merged data does not exist, preparing data')
        hospital_weather = load_data()
    else:
        logging.info('Merged data already exists, loading data')
        hospital_weather = pd.read_feather(DATA_DIR / MERGED_DATA_FILE)
    logging.info(hospital_weather.head())
    # clear out the empty rows
    hospital_weather = prepare_data(hospital_weather)

    logging.info(hospital_weather.head())
    logging.info('Plotting data exploration')
    plot_data_exploration(hospital_weather)

    # Write a metadata file with number of rows, columns, and the column names
    logging.info('Writing metadata file')
    # get min, max, mean and std of each numerical column
    stats = hospital_weather.describe().T
    stats = stats[['min', 'max', 'mean', 'std']]
    stats.reset_index(inplace=True)
    stats.rename(columns={'index': 'column'}, inplace=True)

    with open(DATA_DIR / 'metadata.json', 'w') as f:
        dump(
            {
                'rows': hospital_weather.shape[0],
                'columns': hospital_weather.shape[1],
                'column_names': hospital_weather.columns.tolist(),
                'stats': stats.to_dict(orient='records')
            },
            f
        )
    