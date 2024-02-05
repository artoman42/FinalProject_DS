"""
Script loads the latest trained model, data for inference and predicts results.
"""

import os 
import sys
import pandas as pd
import joblib 
import json
import logging
import time
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)
# print(f"SRC - {SRC_DIR}")
from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
CONF_FILE = "settings.json"
with open(os.path.join(SRC_DIR, CONF_FILE), "r") as file:
    conf = json.load(file)

TEST_DATA_PATH = os.path.join(get_project_dir(conf['general']['processed_data_dir']), conf['inference']['inp_table_name'])
MODELS_DIR_PATH = get_project_dir(conf['general']['models_dir'])
VECTORIZERS_DIR_PATH = get_project_dir(conf['general']['vectorizers_dir'])
PREDICTIONS_DIR_PATH = get_project_dir(conf['general']['predictions_dir'])

if not os.path.exists(PREDICTIONS_DIR_PATH):
    os.makedirs(PREDICTIONS_DIR_PATH)

class Inference():
    def __init__(self):
        self.vectorizer = None
        self.model = None
    def get_vectorizer(self):
        logging.info("Loading vectorizer")
        self.vectorizer = joblib.load(os.path.join(VECTORIZERS_DIR_PATH, conf['general']['vectorizer_name'])) 
    def vectorize_data(self, data):
        logging.info("Vectorizing infer data")
        return self.vectorizer.transform(data)
    def get_latest_model_path(self) -> str:
        """Gets the path of the latest saved model"""
        latest = None
        for (dirpath, dirnames, filenames) in os.walk(MODELS_DIR_PATH):
            for filename in filenames:
                if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pkl') < \
                        datetime.strptime(filename, conf['general']['datetime_format'] + '.pkl'):
                    latest = filename
        return os.path.join(MODELS_DIR_PATH, latest)
    def get_model_by_path(self, path: str):
        """Loads and returns the specified model"""
        try:
            model = joblib.load(path)
            logging.info("Loading model")
            logging.info(f'Path of the model: {path}')
            return model
        except Exception as e:
            logging.error(f'An error occurred while loading the model: {e}')
            sys.exit(1)
    def get_inference_data(self, path: str) -> pd.DataFrame:
        """loads and returns data for inference from the specified csv file"""
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            logging.error(f"An error occurred while loading inference data: {e}")
            sys.exit(1)
    def predict_results(self, model, infer_data):
        logging.info("Predicting results")
        return model.predict(infer_data)
    def get_accuracy(self, predictions, labels):
        return accuracy_score(labels, predictions)
    def store_results(self, infer_df, predictions):
        infer_df["predicted_{}".format(conf['processing']['target_name'])] = predictions
        infer_df.to_csv(os.path.join(PREDICTIONS_DIR_PATH, 'predictions.csv'))
        with open(os.path.join(PREDICTIONS_DIR_PATH, 'metric.txt'), "w") as file:
            file.write(f"Accuracy - {self.get_accuracy(predictions, infer_df[conf['processing']['target_name']])}")
    def run_inference(self):
        logging.info("Running inference")
        start_time = time.time()
        model = self.get_model_by_path(self.get_latest_model_path())

        infer_df = self.get_inference_data(TEST_DATA_PATH)
        infer_text = infer_df[conf['processing']['text_column_name']]
        self.get_vectorizer()
        infer_text = self.vectorize_data(infer_text)
        predictions = self.predict_results(model, infer_text)
        logging.info(f"Accuracy - {self.get_accuracy(predictions, infer_df[conf['processing']['target_name']])}")
        self.store_results(infer_df, predictions)
        end_time = time.time()
        logging.info(f"Inference completed in {end_time - start_time}s")

def main():
    configure_logging()
    inference = Inference()
    inference.run_inference()

if __name__ == "__main__":
    main()