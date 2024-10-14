import subprocess
from utils import *
from src.config.base_config import *
from src.training.annotation import create_tagged_data
from src.training.train import Model
from src.training.predictions import TransformerPredict,getPredictions_spacy
from src.config import logger
import os

# Set up the logger
logs = logger.get_logger(__name__)


def spacy_train(data):
    """
    Trains a spaCy model using the provided training data.

    Args:
    data (list): A list of dictionaries containing the training data.

    Returns:
    str: The output from the last executed command.

    Raises:
    Exception: If any command fails during execution.

    This function performs the following steps:
    1. Creates tagged data for spaCy training.
    2. Initializes the spaCy configuration.
    3. Preprocesses the training and testing data.
    4. Trains the spaCy model using the specified configuration.
    """
    logs.info("Creating folder for training spacy model")
    config_paths = get_paths_instance('spacy')
    create_tagged_data(data,'spacy')


    commands = [
    "python -m spacy init config ./utils/base_config.cfg --lang en --pipeline ner",
    "python -m spacy init fill-config ./utils/base_config.cfg ./utils/config.cfg",
    f"python -m utils.preprocess --train_path \"{config_paths['unique_paths']['training_data_folder']}\" --test_path \"{config_paths['unique_paths']['testing_data_folder']}\"",
    f"python -m spacy train .\\utils\\config.cfg --output \"{config_paths['unique_paths']['model_output_folder']}\" --paths.train \"{config_paths['unique_paths']['training_data_folder']}\\train.spacy\" --paths.dev \"{config_paths['unique_paths']['testing_data_folder']}\\test.spacy\"",
    ]
    # Iterate over each command and execute them one after another
    output = None
    flag = 0
    for command in commands:
        #print(command)
        if(flag == 2):
            logs.info("Training Started...")
        process = subprocess.run(command,capture_output=True, text=True, shell=True)
        if process.returncode != 0:
            logs.error(f"Command '{command}' failed with return code {process.returncode}")
            break
        output = process.stdout
        flag = flag +1
    #logs.info("Training Completed.")
    return output

def spacy_get_entities(input, url):
    """
    Gets named entities from the input text using a trained spaCy model.

    Args:
    input (str): The input text to be processed.
    url (str): The path to the trained spaCy model.

    Returns:
    dict: A dictionary containing the predicted entities and their labels.
    """
    url = os.path.join(url,'model-best')
    if os.path.exists(url):
        return getPredictions_spacy(input,url)
    else:
        return {"Error": "Please give a valid model path"}

def bert_train(url):
    """
    Trains a BERT model using the provided training data.

    Args:
    url (str): The path to the training data.

    Returns:
    dict: A dictionary containing the path to the saved model and the best metrics.

    Raises:
    Exception: If any error occurs during model training.

    This function performs the following steps:
    1. Creates tagged data for BERT training.
    2. Initializes and trains the BERT model.
    3. Saves the trained model and returns the best metrics.
    """
    logs.info("Creating folder for training bert model")
    config_paths = get_paths_instance('bert')
    unique_paths = config_paths['unique_paths']
    create_tagged_data(url,'bert')

    ner_model = Model(unique_paths)

    model_saved_path_name,best_metrics = ner_model.ner_training(model_type=TRF_MODEL_TYPE,model_name=TRF_MODEL_NAME)
    logs.info("Bert Training completed.")
    return {"model_saved_path": model_saved_path_name,"best_model_metris": best_metrics}

def bert_get_entities(input,model_url):
    """
    Gets named entities from the input text using a trained BERT model.

    Args:
    input (str): The input text to be processed.
    model_url (str): The path to the trained BERT model.

    Returns:
    dict: A dictionary containing the input text and the predicted entities.
    """
    url = os.path.join(model_url,'bert_model/best_model')
    if os.path.exists(url):
        trf_pred = TransformerPredict(url)
        prediction = trf_pred.get_prediction(input)
        return {"input": input,"entities": prediction}
    else:
        return {"Error": "Please provide a valid model path"}