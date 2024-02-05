"""
Script for Data processing
"""

import argparse
import os
import sys
import json
import logging
import pandas as pd
import time
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import WordNetLemmatizer
# Define directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)
# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"
with open(os.path.join(SRC_DIR, CONF_FILE), "r") as file:
    conf = json.load(file)
from utils import singleton, get_project_dir, configure_logging

PROCESSED_DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, './data/processed'))
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)
RAW_DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data/raw'))


RAW_TRAIN_PATH = os.path.join(ROOT_DIR, conf['general']['raw_data_dir'], conf['processing']['raw_train_data'])
RAW_TEST_PATH = os.path.join(ROOT_DIR, conf['general']['raw_data_dir'], conf['processing']['raw_test_data'])

PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, conf['train']['table_name'])
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, conf['inference']['inp_table_name'])

parser = argparse.ArgumentParser()
parser.add_argument("--mode",
                    help="Specify data to load training/inference",
                    )

class DataProcessor():
    def __init__(self) -> None:
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        self.STOPWORDS = set(stopwords.words('english'))
        self.PUNCT_TO_REMOVE = string.punctuation
        self.EMOTICONS = {
            u":‑\)":"Happy face or smiley",
            u":\)":"Happy face or smiley",
            u":-\]":"Happy face or smiley",
            u":\]":"Happy face or smiley",
            u":-3":"Happy face smiley",
            u":3":"Happy face smiley",
            u":->":"Happy face smiley",
            u":>":"Happy face smiley",
            u"8-\)":"Happy face smiley",
            u":o\)":"Happy face smiley",
            u":-\}":"Happy face smiley",
            u":\}":"Happy face smiley",
            u":-\)":"Happy face smiley",
            u":c\)":"Happy face smiley",
            u":\^\)":"Happy face smiley",
            u"=\]":"Happy face smiley",
            u"=\)":"Happy face smiley",
            u":‑D":"Laughing, big grin or laugh with glasses",
            u":D":"Laughing, big grin or laugh with glasses",
            u"8‑D":"Laughing, big grin or laugh with glasses",
            u"8D":"Laughing, big grin or laugh with glasses",
            u"X‑D":"Laughing, big grin or laugh with glasses",
            u"XD":"Laughing, big grin or laugh with glasses",
            u"=D":"Laughing, big grin or laugh with glasses",
            u"=3":"Laughing, big grin or laugh with glasses",
            u"B\^D":"Laughing, big grin or laugh with glasses",
            u":-\)\)":"Very happy",
            u":‑\(":"Frown, sad, andry or pouting",
            u":-\(":"Frown, sad, andry or pouting",
            u":\(":"Frown, sad, andry or pouting",
            u":‑c":"Frown, sad, andry or pouting",
            u":c":"Frown, sad, andry or pouting",
            u":‑<":"Frown, sad, andry or pouting",
            u":<":"Frown, sad, andry or pouting",
            u":‑\[":"Frown, sad, andry or pouting",
            u":\[":"Frown, sad, andry or pouting",
            u":-\|\|":"Frown, sad, andry or pouting",
            u">:\[":"Frown, sad, andry or pouting",
            u":\{":"Frown, sad, andry or pouting",
            u":@":"Frown, sad, andry or pouting",
            u">:\(":"Frown, sad, andry or pouting",
            u":'‑\(":"Crying",
            u":'\(":"Crying",
            u":'‑\)":"Tears of happiness",
            u":'\)":"Tears of happiness",
            u"D‑':":"Horror",
            u"D:<":"Disgust",
            u"D:":"Sadness",
            u"D8":"Great dismay",
            u"D;":"Great dismay",
            u"D=":"Great dismay",
            u"DX":"Great dismay",
            u":‑O":"Surprise",
            u":O":"Surprise",
            u":‑o":"Surprise",
            u":o":"Surprise",
            u":-0":"Shock",
            u"8‑0":"Yawn",
            u">:O":"Yawn",
            u":-\*":"Kiss",
            u":\*":"Kiss",
            u":X":"Kiss",
            u";‑\)":"Wink or smirk",
            u";\)":"Wink or smirk",
            u"\*-\)":"Wink or smirk",
            u"\*\)":"Wink or smirk",
            u";‑\]":"Wink or smirk",
            u";\]":"Wink or smirk",
            u";\^\)":"Wink or smirk",
            u":‑,":"Wink or smirk",
            u";D":"Wink or smirk",
            u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
            u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
            u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
            u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
            u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
            u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
            u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
            u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
            u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
            u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
            u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",
            u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
            u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
            u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
            u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
            u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
            u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
            u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
            u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
            u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
            u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
            u":‑\|":"Straight face",
            u":\|":"Straight face",
            u":$":"Embarrassed or blushing",
            u":‑x":"Sealed lips or wearing braces or tongue-tied",
            u":x":"Sealed lips or wearing braces or tongue-tied",
            u":‑#":"Sealed lips or wearing braces or tongue-tied",
            u":#":"Sealed lips or wearing braces or tongue-tied",
            u":‑&":"Sealed lips or wearing braces or tongue-tied",
            u":&":"Sealed lips or wearing braces or tongue-tied",
            u"O:‑\)":"Angel, saint or innocent",
            u"O:\)":"Angel, saint or innocent",
            u"0:‑3":"Angel, saint or innocent",
            u"0:3":"Angel, saint or innocent",
            u"0:‑\)":"Angel, saint or innocent",
            u"0:\)":"Angel, saint or innocent",
            u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
            u"0;\^\)":"Angel, saint or innocent",
            u">:‑\)":"Evil or devilish",
            u">:\)":"Evil or devilish",
            u"\}:‑\)":"Evil or devilish",
            u"\}:\)":"Evil or devilish",
            u"3:‑\)":"Evil or devilish",
            u"3:\)":"Evil or devilish",
            u">;\)":"Evil or devilish",
            u"\|;‑\)":"Cool",
            u"\|‑O":"Bored",
            u":‑J":"Tongue-in-cheek",
            u"#‑\)":"Party all night",
            u"%‑\)":"Drunk or confused",
            u"%\)":"Drunk or confused",
            u":-###..":"Being sick",
            u":###..":"Being sick",
            u"<:‑\|":"Dump",
            u"\(>_<\)":"Troubled",
            u"\(>_<\)>":"Troubled",
            u"\(';'\)":"Baby",
            u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
            u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
            u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
            u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
            u"\(-_-\)zzz":"Sleeping",
            u"\(\^_-\)":"Wink",
            u"\(\(\+_\+\)\)":"Confused",
            u"\(\+o\+\)":"Confused",
            u"\(o\|o\)":"Ultraman",
            u"\^_\^":"Joyful",
            u"\(\^_\^\)/":"Joyful",
            u"\(\^O\^\)／":"Joyful",
            u"\(\^o\^\)／":"Joyful",
            u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",
            u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",
            u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",
            u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",
            u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",
            u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",
            u"\('_'\)":"Sad or Crying",
            u"\(/_;\)":"Sad or Crying",
            u"\(T_T\) \(;_;\)":"Sad or Crying",
            u"\(;_;":"Sad of Crying",
            u"\(;_:\)":"Sad or Crying",
            u"\(;O;\)":"Sad or Crying",
            u"\(:_;\)":"Sad or Crying",
            u"\(ToT\)":"Sad or Crying",
            u";_;":"Sad or Crying",
            u";-;":"Sad or Crying",
            u";n;":"Sad or Crying",
            u";;":"Sad or Crying",
            u"Q\.Q":"Sad or Crying",
            u"T\.T":"Sad or Crying",
            u"QQ":"Sad or Crying",
            u"Q_Q":"Sad or Crying",
            u"\(-\.-\)":"Shame",
            u"\(-_-\)":"Shame",
            u"\(一一\)":"Shame",
            u"\(；一_一\)":"Shame",
            u"\(=_=\)":"Tired",
            u"\(=\^\·\^=\)":"cat",
            u"\(=\^\·\·\^=\)":"cat",
            u"=_\^=	":"cat",
            u"\(\.\.\)":"Looking down",
            u"\(\._\.\)":"Looking down",
            u"\^m\^":"Giggling with hand covering mouth",
            u"\(\・\・?":"Confusion",
            u"\(?_?\)":"Confusion",
            u">\^_\^<":"Normal Laugh",
            u"<\^!\^>":"Normal Laugh",
            u"\^/\^":"Normal Laugh",
            u"\（\*\^_\^\*）" :"Normal Laugh",
            u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
            u"\(^\^\)":"Normal Laugh",
            u"\(\^\.\^\)":"Normal Laugh",
            u"\(\^_\^\.\)":"Normal Laugh",
            u"\(\^_\^\)":"Normal Laugh",
            u"\(\^\^\)":"Normal Laugh",
            u"\(\^J\^\)":"Normal Laugh",
            u"\(\*\^\.\^\*\)":"Normal Laugh",
            u"\(\^—\^\）":"Normal Laugh",
            u"\(#\^\.\^#\)":"Normal Laugh",
            u"\（\^—\^\）":"Waving",
            u"\(;_;\)/~~~":"Waving",
            u"\(\^\.\^\)/~~~":"Waving",
            u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
            u"\(T_T\)/~~~":"Waving",
            u"\(ToT\)/~~~":"Waving",
            u"\(\*\^0\^\*\)":"Excited",
            u"\(\*_\*\)":"Amazed",
            u"\(\*_\*;":"Amazed",
            u"\(\+_\+\) \(@_@\)":"Amazed",
            u"\(\*\^\^\)v":"Laughing,Cheerful",
            u"\(\^_\^\)v":"Laughing,Cheerful",
            u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",
            u'\(-"-\)':"Worried",
            u"\(ーー;\)":"Worried",
            u"\(\^0_0\^\)":"Eyeglasses",
            u"\(\＾ｖ\＾\)":"Happy",
            u"\(\＾ｕ\＾\)":"Happy",
            u"\(\^\)o\(\^\)":"Happy",
            u"\(\^O\^\)":"Happy",
            u"\(\^o\^\)":"Happy",
            u"\)\^o\^\(":"Happy",
            u":O o_O":"Surprised",
            u"o_0":"Surprised",
            u"o\.O":"Surpised",
            u"\(o\.o\)":"Surprised",
            u"oO":"Surprised",
            u"\(\*￣m￣\)":"Dissatisfied",
            u"\(‘A`\)":"Snubbed or Deflated"
        }
        self.lemmatizer=WordNetLemmatizer()

    def get_raw_data(self, path:str) -> pd.DataFrame:
        """function to get raw data"""
        logging.info("Getting raw data")
        return pd.read_csv(path)
    def remove_stopwords(self, text):
        """custom function to remove the stopwords"""
        return " ".join([word for word in word_tokenize(text) if word.lower() not in self.STOPWORDS])
    def make_lower_casing(self, text):
        """custom function to lower text"""
        return text.lower()
    def remove_punctuation(self, text, PUNCT_TO_REMOVE):
        """custom function to remove the punctuation"""
        return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    def get_word_counter(self, df):
        """custom function to get word counts"""
        cnt = Counter()
        for text in df["review"].values:
            for word in text.split():
                cnt[word] += 1
        return cnt
    def get_most_frequent_words(self, df, amount):
        """custom function to get most frequent words"""
        return set([w for (w, wc) in self.get_word_counter(df).most_common(amount)])
    def get_most_rare_words(self, df, amount):
        """custom function to get most rare words"""
        return set([w for (w, wc) in self.get_word_counter(df).most_common()[:-amount-1:-1]])
    def remove_words(self, text, words_to_remove):
        """custom function to remove the frequent words"""
        return " ".join([word for word in str(text).split() if word not in words_to_remove])
    def remove_urls(self, text):
        """custom function to remove urls from text"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    def remove_html(self, text):
        """custom function to remove html tags from text"""
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)
    def remove_emoji(self, string):
        """custom function to remove emojis"""
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)
    def remove_emoticons(self, text):
        """custom function to remove emoticons"""
        emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in self.EMOTICONS) + u')')
        return emoticon_pattern.sub(r'', text)
    def save_data(self, df:pd.DataFrame, path_to_save:str):
        """function to save data"""
        logging.info("Saving data")
        df.to_csv(path_to_save)
    def lemmatize_words(self,words):
        """custom function to lemmatize words"""
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(word for word in words)
    def run_pipeline(self, raw_path:str, path_to_save:str, 
                     text_column_name:str) -> pd.DataFrame:
        logging.info("Starting preprocessing pipeline...")
        start_time = time.time()
        df = self.get_raw_data(raw_path)
        logging.info("Removing stopwords")
        df[text_column_name] = df[text_column_name].apply(lambda text: self.remove_stopwords(text))
        logging.info("Making lower casing")
        df[text_column_name] = df[text_column_name].apply(self.make_lower_casing)
        logging.info("Removing punctuation")
        df[text_column_name] = df[text_column_name].apply(lambda text: self.remove_punctuation(text,
                                                                                                self.PUNCT_TO_REMOVE))
        # logging.info("Removing 10 most frequent and most rare words")
        # df[text_column_name] = df[text_column_name].apply(lambda text: 
        #                                                   self.remove_words(text,
        #                                                  [self.get_most_frequent_words(df, 10),
        #                                                    self.get_most_rare_words(df, 10)]))
        logging.info("Removing urls")
        df[text_column_name] = df[text_column_name].apply(self.remove_urls)
        logging.info("Removing HTML")
        df[text_column_name] = df[text_column_name].apply(self.remove_html)
        logging.info("Removing emojis")
        df[text_column_name] = df[text_column_name].apply(self.remove_emoji)
        logging.info("Removing emoticons")
        df[text_column_name] = df[text_column_name].apply(self.remove_emoticons)
        df[text_column_name] = df[text_column_name].apply(lambda text: word_tokenize(text))
        logging.info("Lemmatizing words")
        df[text_column_name] = df[text_column_name].apply(self.lemmatize_words)
        self.save_data(df, path_to_save)
        end_time=time.time()
        logging.info(f"Preprocessing pipeline done and results saved. Completed in {end_time-start_time}.")
    
def main():
    configure_logging()
    data_proc = DataProcessor()
    args = parser.parse_args()
    if args.mode == "training":
        data_proc.run_pipeline(RAW_TRAIN_PATH, PROCESSED_TRAIN_PATH, conf['processing']['text_column_name'])
    elif args.mode == "inference":
        data_proc.run_pipeline(RAW_TEST_PATH, PROCESSED_TEST_PATH, conf['processing']['text_column_name'])
    else:
        logging.info("Bad mode exception, check args to command")

if __name__ == "__main__":
    main()