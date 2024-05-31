import logging
from pathlib import Path
import sys

# add directory of the current script to the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from hospital_weather.explore import explore_data



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    explore_data()
