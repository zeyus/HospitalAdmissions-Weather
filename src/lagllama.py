import logging
from pathlib import Path
import sys
import argparse

# add directory of the current script to the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from hospital_weather.LagLlama import generate_lagllama_forecasts, calculate_prediction_rmse



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rmse", action="store_true")
    args = parser.parse_args()
    if args.rmse:
        logging.info("Calculating RMSE for Lag Llama predictions")
        rmse = calculate_prediction_rmse()
        logging.info(f"RMSE: {rmse}")
    else:
        logging.info("Generating Lag Llama predictions")
        generate_lagllama_forecasts()
