'''
Script to upload data for training and testing
'''

# Importing required libraries
import logging
import os
import sys
import json
import requests
from zipfile import ZipFile
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)
print(f"ROOT - {ROOT_DIR}")
from utils import singleton, get_project_dir, configure_logging

RAW_DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, './data/raw'))
if not os.path.exists(RAW_DATA_DIR):
    os.makedirs(RAW_DATA_DIR)
print(f"Raw data dir - {ROOT_DIR}")
# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
RAW_DATA_DIR = get_project_dir(conf['general']['raw_data_dir'])
# RAW_TRAIN_PATH = os.path.join(RAW_DATA_DIR, conf['train']['raw_table_name'])
# RAW_INFERENCE_PATH = os.path.join(RAW_DATA_DIR, conf['inference']['raw_inp_table_name'])
TRAIN_DATA_LINK = conf['general']['train_data_link']
TEST_DATA_LINK = conf['general']['test_data_link']

@singleton
class DataLoader():
    def __init__(self):
        pass
    def download_and_extract_zip(self, url, destination_folder):
        """
        Download a zip file from the given URL and extract its contents to the destination folder.
        """
        # Ensure the destination folder exists
        os.makedirs(destination_folder, exist_ok=True)

        # Download the zip file
        response = requests.get(url)
        zip_file_path = os.path.join(destination_folder, "downloaded_file.zip")

        with open(zip_file_path, "wb") as zip_file:
            zip_file.write(response.content)

        # Extract the contents of the zip file
        with ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(destination_folder)

        # Remove the downloaded zip file
        os.remove(zip_file_path)

if __name__ == "__main__":
    configure_logging()
    logger.info("Starting DataLoading script...")
    dataLoader = DataLoader()
    dataLoader.download_and_extract_zip(TRAIN_DATA_LINK, RAW_DATA_DIR)
    dataLoader.download_and_extract_zip(TEST_DATA_LINK, RAW_DATA_DIR)
    logger.info("DataLoading completed successfully.")