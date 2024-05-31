import logging
from pathlib import Path
import sys

# add directory of the current script to the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from hospital_weather.LinearRegression import train_lr



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_lr()
