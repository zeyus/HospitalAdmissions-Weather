# Code adapted from: https://colab.research.google.com/drive/1DRAzLUPxsd-0r8b-o4nlyFXrjw_ZajJJ?usp=sharing#scrollTo=vPBAO18DWT8A
# https://github.com/time-series-foundation-models/lag-llama


from itertools import islice
import logging

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator  # type: ignore
from gluonts.dataset.repository.datasets import get_dataset  # type: ignore

from gluonts.dataset.pandas import PandasDataset  # type: ignore
import pandas as pd
from tqdm import tqdm

from common.config import DATA_DIR, FIGURE_DIR, read_data, prepare_data, MODEL_DIR

# add contrib/lag-llama to the path
import sys
sys.path.append("contrib/lag-llama")
from lag_llama.gluon.estimator import LagLlamaEstimator


def get_lag_llama_predictions(dataset, prediction_length, context_length=32, num_samples=20, device="cuda", batch_size=64, nonnegative_pred_samples=True):
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
        },

        batch_size=batch_size,
        num_parallel_samples=num_samples,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
    tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))

    return forecasts, tss


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')
    # Load the data
    logging.info('Loading data')
    data = read_data()
    data, _ = prepare_data(data)
    

    target = 'COVID-19 admissions (suspected and confirmed)'
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
    data['item_id'] = "Covid Admissions"

    print(data.head())
    data['target'] = data['target'].astype('float32')

    # Create a PandasDataset from the data
    logging.info('Creating dataset')
    dataset = PandasDataset.from_long_dataframe(data, target="target", item_id="item_id")

    prediction_length = 7
    context_length = 8
    num_samples = 100

    # Get the forecasts
    logging.info('Getting forecasts')
    forecasts, tss = get_lag_llama_predictions(dataset, prediction_length, context_length=context_length, num_samples=num_samples, device=device)

    # Plot the forecasts
    logging.info('Plotting forecasts')
    plt.figure(figsize=(20, 15))
    date_formater = mdates.DateFormatter('%b, %d')
    plt.rcParams.update({'font.size': 15})
    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
        ax = plt.subplot(3, 3, idx + 1)
        
        plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target", )
        forecast.plot( color='g')
        plt.xticks(rotation=60)
        ax.xaxis.set_major_formatter(date_formater)
        ax.set_title(forecast.item_id)

    plt.gcf().tight_layout()
    plt.legend()

    logging.info('Saving figure')
    plt.savefig(FIGURE_DIR / "lag_llama_predictions.png")
