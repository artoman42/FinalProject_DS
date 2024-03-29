"""
This script vectorizes data, trains model, and save it to pickle file.
"""

import os
import sys
import json
import pandas as pd
import datetime
import logging
import time
import joblib
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Defing root and src dirs
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
CONF_FILE = "settings.json"
with open(os.path.join(SRC_DIR, CONF_FILE), "r") as file:
    conf = json.load(file)

#defining pathes
TRAIN_DATA_PATH = os.path.join(get_project_dir(conf['general']['processed_data_dir']), conf['train']['table_name'])
MODELS_DIR_PATH = get_project_dir(conf['general']['models_dir'])
VECTORIZERS_DIR_PATH = get_project_dir(conf['general']['vectorizers_dir'])

if not os.path.exists(MODELS_DIR_PATH):
    os.makedirs(MODELS_DIR_PATH)
if not os.path.exists(VECTORIZERS_DIR_PATH):
    os.makedirs(VECTORIZERS_DIR_PATH)
#main class of trainig
class Training():
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer(min_df = 10, token_pattern = r'[a-zA-Z]+')
        self.model = LogisticRegression(C=conf['train']['C'],
                                        max_iter=conf['train']['max_iter'],
                                        penalty=conf['train']['penalty'],
                                        solver=conf['train']['solver'])
    def vectorize_data(self):
        """function to vectorize data"""
        logging.info("Vectorizing text data")
        return self.vectorizer.fit_transform(self.df[conf['processing']['text_column_name']])
    
    def save_vectorizer(self):
        """function to save vectorizer in pickle file"""
        logging.info("Saving Vectorizer")
        logging.info(f"Vectorizer name - {conf['general']['vectorizer_name']}")
        joblib.dump(self.vectorizer,
                    os.path.join(VECTORIZERS_DIR_PATH,
                                   conf['general']['vectorizer_name']))
    
    def train_model(self, X, y):
        """function to train model"""
        logging.info("Training model")
        self.model.fit(X, y )
    
    def save_model(self):
        """function to save model"""
        logging.info("Saving model...")
        model_name = datetime.now().strftime(conf['general']['datetime_format']) + '.pkl'
        logging.info(f"Model name - {model_name}")
        joblib.dump(self.model,
                     os.path.join(MODELS_DIR_PATH, 
                                  model_name))
        
    def show_accuracy_on_sample(self,X, sample_size):
        """custom function to show achieved accuracy 
        on first{sample_size} raws of traing data"""
        predictions = self.model.predict(X[:sample_size])
        true_labels = self.df[conf['processing']['target_name']].values[:sample_size]
        return accuracy_score(true_labels, predictions)
    
    def run_training(self):
        """function to run training process"""
        logging.info("Starting training")
        start_time = time.time()
        X = self.vectorize_data()
        self.save_vectorizer()
        self.train_model(X, self.df[conf['processing']['target_name']])
        self.save_model()
        end_time = time.time()
        logging.info(f"Training completed in {end_time-start_time}s")
        logging.info(f"Accuracy on 1000 samples - {self.show_accuracy_on_sample(X, 1000)}")

def main():
    configure_logging()
    training = Training(TRAIN_DATA_PATH)
    training.run_training()
if __name__ == "__main__":
    main()