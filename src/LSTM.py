import logging
from pathlib import Path
import sys
import argparse

# add directory of the current script to the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from hospital_weather.LSTM import train_lstm, compare_lstm_models



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        logging.info("Comparing LSTM models")
        compare_lstm_models()
    else:
        logging.info("Training LSTM")
        train_lstm()
