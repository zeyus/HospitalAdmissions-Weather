# Forecasting COVID-19 Related Hospital Admissions Using Weather Data

![a representation of a coronavirus particles](https://github.com/user-attachments/assets/8f96c6c0-dfee-4bba-bb94-9264bbfa7d23)

The COVID-19 pandemic put a strain on the global healthcare infrastructure, the ability to forecast an influx of patients can assist healthcare systems to be prepared in the case of another pandemic or epidemic. This study investigates the use of linear regression, Long Short-Term Memory (LSTM) and Lag-Llama models for predicting the number of daily patient admissions in the country of Wales during the COVID-19 pandemic using local weather data. Models were evaluated for performance with the Root Mean Squared Error (RMSE) score on a set of test data, resulting in an RMSE of 5.84 for the LSTM, 7.37 for Lag-Llama and 33.65 for linear regression. The results indicate that LSTMs are a promising architecture for time-series forecasting, but further investigation is needed, especially for the validation of the magnitude of weather’s effect on COVID-19 hospitalizations.

## Setup

Clone this repo and install the requirements with pip `pip install -r requirements.txt`

### Data

The data used is the daily COVID-19 hospital admissions in Wales from:

> Office for National Statistics. (2024). Standard Area Measurements (Latest) for Administrative Areas in the United Kingdom (No. 27700) [Dataset]. https://geoportal.statistics.gov.uk/datasets/ons::standard-area-measurements-latest-for-administrative-areas-in-the-united-kingdom/about

Weather data were sourced from Visual Crossing

> Visual Crossing Corporation. (2024). Visual Crossing Weather (2020-2023) [Data service]

The hourly data were used from multiple weather stations for Wales weather during the period covered in the COVID-19 dataset.

It should be fairly trivial to adjust the code to support other datasets.

## Usage

There are a few helper scripts in the `src` folder.

- `src/prepare.py` : prepare the source data and plot some visual exploration
- `src/lr.py` : Run the linear regression model and save plots / stats
- `src/LSTM.py [-c]` : train the LSTM model. If the `-c` argument is provided, instead, compare the saved LSTM models.
- `src/lagllama.py` : Run lagllama on the data and save forecasts + plots

All LSTM hyperparameters can be configured in [`src/hospital_weather/LSTM.py#L151`](https://github.com/zeyus/HospitalAdmissions-Weather/blob/main/src/hospital_weather/LSTM.py#L151)

## LSTM Architecture

The final model is capable of handling arbitrary batch sizes and input features (as long as the number of features doesn't change for a particular model implementation).

The architecture is outlined in the following diagram:

![image](https://github.com/user-attachments/assets/c5580b62-8066-4e9e-a2c3-8b725939d005)


## Sample predictions

### Linear Regression Model

![image](https://github.com/user-attachments/assets/e4cf20d7-7fe5-4d69-a844-a626ae83c794)

### LSTM model

![image](https://github.com/user-attachments/assets/c7c949f6-6933-448b-bd43-b39df02744eb)

### Lag-Lama (Zero-Shot, no fine-tuning)

![image](https://github.com/user-attachments/assets/aa15c9e8-6d23-47b5-815c-1700c9df991a)
