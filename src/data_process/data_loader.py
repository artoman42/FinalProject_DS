"""Script to upload data for training and testing"""

# Importing required libraries
import argparse
import logging
import os
import sys
import json
import requests
import zipfile
from zipfile import ZipFile
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

from utils import singleton, get_project_dir, configure_logging

RAW_DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, './data/raw'))
if not os.path.exists(RAW_DATA_DIR):
    os.makedirs(RAW_DATA_DIR)
# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(os.path.join(SRC_DIR, CONF_FILE), "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
RAW_DATA_DIR = get_project_dir(conf['general']['raw_data_dir'])
TRAIN_DATA_LINK = conf['general']['train_data_link']
TEST_DATA_LINK = conf['general']['test_data_link']

parser = argparse.ArgumentParser()
parser.add_argument("--mode",
                    help="Specify data to load training/inference",
                    )
@singleton
class DataLoader():
    def __init__(self):
        pass
    def download_and_extract_zip(self, url, destination_folder):
        """
        Download a zip file from the given URL and extract its contents to the destination folder.
        """
        os.makedirs(destination_folder, exist_ok=True)

        response = requests.get(url)
        zip_file_path = os.path.join(destination_folder, "downloaded_file.zip")

        with open(zip_file_path, "wb") as zip_file:
            zip_file.write(response.content)

        with ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(destination_folder)

        os.remove(zip_file_path)

if __name__ == "__main__":
    configure_logging()
    logger.info("Starting DataLoading script...")
    args = parser.parse_args()
    dataLoader = DataLoader()
    if args.mode == "training":
        dataLoader.download_and_extract_zip(TRAIN_DATA_LINK, RAW_DATA_DIR)
    elif args.mode == "inference":
        dataLoader.download_and_extract_zip(TEST_DATA_LINK, RAW_DATA_DIR)
    else:
        logger.info("Bad mode exception, check args to command")
    logger.info("DataLoading completed successfully.")