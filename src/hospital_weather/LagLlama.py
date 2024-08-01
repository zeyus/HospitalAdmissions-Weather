# Code adapted from: https://colab.research.google.com/drive/1DRAzLUPxsd-0r8b-o4nlyFXrjw_ZajJJ?usp=sharing#scrollTo=vPBAO18DWT8A
# https://github.com/time-series-foundation-models/lag-llama


import logging

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator  # type: ignore
from gluonts.dataset.repository.datasets import get_dataset  # type: ignore

from gluonts.dataset.pandas import PandasDataset  # type: ignore
from tqdm import tqdm

from json import dumps


import sys
# add contrib/lag-llama to the path
sys.path.append("contrib/lag-llama")
# add directory of the current script to the path
sys.path.append("src")


from hospital_weather.common.config import DATA_DIR, FIGURE_DIR, read_data, prepare_data


from lag_llama.gluon.estimator import LagLlamaEstimator


def get_lag_llama_predictions(
        dataset: PandasDataset,
        prediction_length: int,
        context_length: int = 32,
        num_samples: int = 20,
        device: str = "cuda",
        batch_size: int = 128,
        nonnegative_pred_samples: bool = True,
        progress: bool = False,
        num_parallel_samples: int = 1000,
        rope_scaling: bool = False):
    ckpt = torch.load("./contrib/hf_model/lag-llama.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="./contrib/hf_model/lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],

        nonnegative_pred_samples=nonnegative_pred_samples,

        # linear positional encoding scaling
        rope_scaling={
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        } if rope_scaling else None,

        batch_size=batch_size,
        num_parallel_samples=num_parallel_samples,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    if progress:
        forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
        tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))
    else:
        forecasts = list(forecast_it)
        tss = list(ts_it)

    return forecasts, tss


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f'Using device: {device}')
    # Load the data
    logging.info('Loading data')
    data = prepare_data(read_data())

    target = 'cov19'
    features = [
        target,
        # 'day_of_week',
        # 'humidity',
        # 'temp',
        # 'windspeed',
        # 'winddir'
    ]

    data = data[features]
    data = data.rename(columns={target: 'target'})
    
    # Add a pseudo item_id column
    # data['item_id'] = "Covid Admissions"

    # add an index to item_id
    # data['item_id'] = data['item_id'].astype('category')

    print(data.head())
    data['target'] = data['target'].astype('float32')
    # data['day_of_week'] = data['day_of_week'].astype('float32')
    # data['humidity'] = data['humidity'].astype('float32')
    # data['temp'] = data['temp'].astype('float32')
    # data['windspeed'] = data['windspeed'].astype('float32')
    # data['winddir'] = data['winddir'].astype('float32')
    print(data.head())

    # Create a PandasDataset from the data
    logging.info('Creating dataset')
    dataset = PandasDataset(
        data[["target"]],
        target="target",
        assume_sorted=True,
        # unchecked=True,
        freq="D",
        # static_features=data.drop(columns=["target"]),
    )

    prediction_length = 2
    context_length = 14
    num_samples = 500

    # Get the forecasts
    logging.info('Getting forecasts')
    forecasts, tss = get_lag_llama_predictions(dataset, prediction_length, context_length=context_length, num_samples=num_samples, device=device)

    # Plot the forecasts
    logging.info('Plotting forecasts')
    plt.figure(figsize=(20, 15))
    date_formater = mdates.DateFormatter('%b, %d')
    plt.rcParams.update({'font.size': 15})
    for idx, (forecast, ts) in enumerate(zip(forecasts, tss)):
        ax = plt.subplot(1, 1, idx + 1)
        
        plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target", )
        forecast.plot( color='g')
        plt.xticks(rotation=60)
        ax.xaxis.set_major_formatter(date_formater)
        ax.set_title(target)

    plt.gcf().tight_layout()
    plt.legend()

    logging.info('Saving figure')
    plt.savefig(FIGURE_DIR / "lag_llama_predictions.png")

    logging.info('Done!')

    # now try windowed predictions
    logging.info('Generating windowed forecasts')
    num_rows = len(data)
    window_size = context_length + prediction_length
    step_size = prediction_length
    forecasts = []
    for i in tqdm(range(0, num_rows - window_size, step_size)):
        dataset_window = PandasDataset(
            data[i:i + window_size],
            target="target",
            assume_sorted=True,
            freq="D",
        )
        forecast, _ = get_lag_llama_predictions(dataset_window, prediction_length, context_length=context_length, num_samples=num_samples, device=device)
        forecasts.append(forecast)
    
    # make a dataframe with the forecasts
    logging.info('Creating dataframe with windowed forecasts')
    windowed_forecasts = data.copy()
    windowed_forecasts['forecast_mean'] = np.array([np.nan] * windowed_forecasts.shape[0], dtype='float32')
    windowed_forecasts['forecast_lower'] = np.array([np.nan] * windowed_forecasts.shape[0], dtype='float32')
    windowed_forecasts['forecast_upper'] = np.array([np.nan] * windowed_forecasts.shape[0], dtype='float32')
    windowed_forecasts['samples'] = [""] * windowed_forecasts.shape[0]
    for i, forecast in enumerate(forecasts):
        indices = forecast[0].index.to_timestamp()
        windowed_forecasts.loc[indices, 'forecast_mean'] = forecast[0].mean.tolist()
        windowed_forecasts.loc[indices, 'forecast_lower'] = np.quantile(forecast[0].samples, 0.05, axis=0).tolist()
        windowed_forecasts.loc[indices, 'forecast_upper'] = np.quantile(forecast[0].samples, 0.95, axis=0).tolist()
        for i in range(prediction_length):
            windowed_forecasts.loc[indices[i], 'samples']= dumps(forecast[0].samples[:, i].tolist())
    
    # Plot the forecasts
    logging.info('Plotting windowed forecasts')
    plt.figure(figsize=(20, 15))
    date_formater = mdates.DateFormatter('%b, %d')
    plt.rcParams.update({'font.size': 15})
    ax = plt.subplot(1, 1, 1)
    plt.plot(windowed_forecasts['target'], label="target", )
    plt.plot(windowed_forecasts['forecast_mean'], label="forecast", color='g')
    plt.fill_between(
        windowed_forecasts.index,
        windowed_forecasts['forecast_lower'],
        windowed_forecasts['forecast_upper'],
        where=(windowed_forecasts['forecast_mean'] != None).to_list(),
        color='g',
        alpha=0.3)
    plt.xticks(rotation=60)
    ax.xaxis.set_major_formatter(date_formater)
    ax.set_title(target)

    plt.gcf().tight_layout()
    plt.legend()

    logging.info('Saving figure')
    plt.savefig(FIGURE_DIR / "lag_llama_windowed_predictions.png")

    logging.info('Done!')

    # Save the forecasts
    windowed_forecasts.to_csv(DATA_DIR / 'lag_llama_windowed_forecasts.csv')
    logging.info('Done!')
