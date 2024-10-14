#With Exception Handling
import spacy
from spacy.tokens import DocBin
from src.config import logger
# from check import config_paths

import pickle
import os
import argparse


# Set up the logger
logs = logger.get_logger(__name__)

nlp = spacy.blank("en")

# config_paths = get_paths_instance()
def load_data(file_path):
    logs.info("Getting path instance")
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError as e:
        logs.error(f"File not found: {file_path}")
        raise e
    except pickle.UnpicklingError as e:
        logs.error(f"Error unpickling file: {file_path}")
        raise e
    except Exception as e:
        logs.error(f"An error occurred: {e}")
        raise e
        #return None

# Load Data
def prepare_spacy_data(train_path,test_path):

    logs.info("Loading Data...")
    train_data_path = os.path.join(train_path,'TrainData.pickle')
    test_data_path = os.path.join(test_path, 'TestData.pickle')

    # print(f"Train_data={train_data_path}")
    training_data = load_data(train_data_path)
    testing_data = load_data(test_data_path)

    if training_data is not None:
        # the DocBin will store the example documents
        db = DocBin()
        for _ , text, annotations in training_data:
            doc = nlp(text)
            ents = []
            for start, end, label in annotations['entities']:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)
        try:
            db.to_disk(os.path.join(train_path, 'train.spacy'))
        except Exception as e:
            logs.error(f"An error occurred while saving train data: {e}")
            raise e

    if testing_data is not None:
    # the DocBin will store the example documents
        db_test = DocBin()
        for _, text, annotations in testing_data:
            doc = nlp(text)
            ents = []
            for start, end, label in annotations['entities']:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    ents.append(span)
            doc.ents = ents
            db_test.add(doc)
        try:
            db_test.to_disk(os.path.join(test_path, 'test.spacy'))
        except Exception as e:
            logs.error(f"An error occurred while saving test data: {e}")
            raise e

def arg_parser():
    logs.info("Parsing text.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str,required=True)
    parser.add_argument("--test_path", type=str,required=True)

    args = parser.parse_args()

    return args

def main():
    cmd_args = arg_parser()
    if cmd_args.train_path and cmd_args.test_path:
        prepare_spacy_data(cmd_args.train_path,cmd_args.test_path)


if __name__=="__main__":
    main()
    logs.info("Preprocessing completed.")