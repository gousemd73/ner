import json
import numpy as np
import pandas as pd
import re
import string
import random
import pickle
import os
from src.config import logger
from src.config.base_config import (data_tagged_text,overall_tagged_data,
overall_untagged_data,trf_train_data,trf_test_data,get_paths_instance)

# Set up the logger
logs = logger.get_logger(__name__)

def load_training_data(file_path):
    """
    Loads training data from a JSON file.

    Args:
    file_path (str): The path to the JSON file containing the training data.

    Returns:
    dict: The training data loaded from the JSON file.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    JSONDecodeError: If the file is not a valid JSON.
    """
    logs.info("Training Data Loading...")
    try:
        with open(file_path) as file:
            return json.load(file)
    except FileNotFoundError:
        logs.error(f"Error: The file {file_path} was not found.")
        raise FileNotFoundError
    except json.JSONDecodeError:
        logs.error("Error: Failed to decode JSON from the file.")
        raise json.JSONDecodeError

def clean_text(text):
    """
    Cleans the input text by performing various text preprocessing steps.

    Args:
    text (str): The text to be cleaned.

    Returns:
    str: The cleaned text.

    Raises:
    Exception: If any error occurs during text cleaning.
    """
    try:
        #logs.info("text preprocessing.")
        text = text.lower()
        text = re.sub(r"[$]([\d]+[,][\d]+)+", lambda match: match.group().replace(",",""), text)
        text = text.replace("-"," ")
        text = text.replace('\n', ' ')
        text = text.replace('(', '')
        text = text.replace(')', '')
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)

        return text
    except Exception as e:
        logs.error(f"Error cleaning text: {e}")
        raise e

def split_text(text):
    """
    Splits the input text into a list of words or phrases.

    Args:
    text (str): The text to be split.

    Returns:
    list: A list of words or phrases obtained from the input text.

    Raises:
    Exception: If any error occurs during text splitting.
    """
    #logs.info("Text spliting")
    try:
        dataList = re.split("-| ", str(text))
        dataList = [sentence for sentence in dataList if sentence.strip()]
        return dataList
    except Exception as e:
        logs.error(f"Error splitting text: {e}")
        raise e

def create_dataframe(training_data):
    """
    Creates a DataFrame from the training data.

    Args:
    training_data (list): A list of dictionaries containing the training data.

    Returns:
    DataFrame: A pandas DataFrame containing the cleaned and split text data.

    Raises:
    Exception: If any error occurs during DataFrame creation.
    """
    logs.info("Creating dataframe")
    all_medical_text = pd.DataFrame(columns=['id', 'text', 'tags'])
    try:

        for i in range(len(training_data)):
            # print(training_data[i]['text'])
            cleaned_text = clean_text(training_data[i]['text'])
            data_list = split_text(cleaned_text)
            df = pd.DataFrame(data_list, columns=['text'])
            df['id'] = i
            # print(data)
            all_medical_text = pd.concat((all_medical_text, df), ignore_index=True)
    except Exception as e:
        logs.error(f"Error creating DataFrame: {e}")
        raise e
        # print(all_medical_text)
    return all_medical_text

def save_to_excel(df, file_path):
    """
    Saves the DataFrame to an Excel file.

    Args:
    df (DataFrame): The DataFrame to be saved.
    file_path (str): The path to the Excel file.

    Raises:
    Exception: If any error occurs during saving to Excel.
    """
    #logs.info("Save dataframe into excel file.")
    try:
        df.to_excel(file_path, index=False)
    except Exception as e:
        logs.error(f"Error saving to Excel: {e}")
        raise e

def create_entity_dict(training_data):
    """
    Creates a dictionary of entities from the training data.

    Args:
    training_data (list): A list of dictionaries containing the training data,
    where each dictionary has an 'entities' key with nested dictionaries or lists.

    Returns:
    tuple: A tuple containing:
    - dict: A dictionary where each key is an index corresponding to the training data entry,
    and each value is a dictionary of entities with their processed values.
    - set: A set of unique keys found in the entities.

    Raises:
    Exception: If any error occurs during the creation of the entity dictionary.
    """
    logs.info("Create entity dictionary")
    try:
        dicts = {}
        unique_keys=set()
        for j in range(len(training_data)):
            dictss = {}
            for key, value in training_data[j]['entities'].items():
                if isinstance(value, dict):
                # Handle dictionary values
                    for sub_key, sub_value in value.items():
                        sub_value = str(sub_value).lower()
                        if sub_key == "location":
                            sub_value = sub_value.replace(",", " ")
                        sub_value=re.sub(r"[-\n().]", "", sub_value)
                        x = re.split(",", str(sub_value))
                        x = [sentence for sentence in x if sentence.strip()]
                        for i in range(len(x)):
                            p = x[i].replace(":", "")
                            p = re.split("-| ", str(p))
                            p = [sentences for sentences in p if sentences.strip()]
                            if sub_key in dictss:
                                dictss[sub_key].append(p)
                            else:
                                dictss[sub_key] = [p]
                            unique_keys.update([sub_key])
                elif isinstance(value, list):
                    # Handle list of dictionaries
                    for item in value:
                        if isinstance(item, dict):
                            for sub_key, sub_value in item.items():
                                sub_value = str(sub_value).lower()
                                if sub_key == "location":
                                    sub_value = sub_value.replace(",", " ")
                                sub_value=re.sub(r"[-\n().]", " ", sub_value)
                                x = re.split(",", str(sub_value))
                                x = [sentence for sentence in x if sentence.strip()]
                                for i in range(len(x)):
                                    p = x[i].replace(":", "")
                                    p = re.split("-| ", str(p))
                                    p = [sentences for sentences in p if sentences.strip()]
                                    if sub_key in dictss:
                                        dictss[sub_key].append(p)
                                    else:
                                        dictss[sub_key] = [p]
                                    unique_keys.update([sub_key])
                else:
                    # Handle string values
                    value = str(value).lower()
                    if key == "location":
                        value = value.replace(",", " ")
                    value=re.sub(r"[-\n().]", " ", value)
                    x = re.split(",", str(value))
                    x = [sentence for sentence in x if sentence.strip()]
                    for i in range(len(x)):
                        p = x[i].replace(":", "")
                        p = re.split("-| ", str(p))
                        p = [sentences for sentences in p if sentences.strip()]
                        if key in dictss:
                            dictss[key].append(p)
                        else:
                            dictss[key] = [p]
                    unique_keys.update([key])
            dicts[j] = dictss
        return dicts, unique_keys
    except Exception as e:
        logs.error(f"Error creating entity dictionary: {e}")
        raise e


def add_tag_in_csv(df, i, key, value, count,unique_labels):
    """
    Adds a tag to the DataFrame based on the entity type and position.

    Args:
    df (DataFrame): The DataFrame to update.
    i (int): The index of the current entity in the DataFrame.
    key (str): The entity type.
    value (str): The text value to tag.
    count (int): The position of the value in the entity list.
    unique_labels (set): A set of unique entity labels.

    Raises:
    Exception: If any error occurs while adding the tag.
    """
    try:
        if key in unique_labels and count == 0:
            df.loc[(df['id'] == i) & (df['text'] == value), 'tags'] = 'B-'+ key
        else:
            df.loc[(df['id'] == i) & (df['text'] == value), 'tags'] = 'I-' + key

    except Exception as e:
        logs.error(f"Error adding tag in CSV: {e}")
        raise e

def tag_data(df, entity_dict,unique_labels):
    """
    Tags the DataFrame with entity labels based on the provided entity dictionary.

    Args:
    df (DataFrame): The DataFrame to tag.
    entity_dict (dict): A dictionary of entities to tag in the DataFrame.
    unique_labels (set): A set of unique entity labels.

    Returns:
    DataFrame: The tagged DataFrame.

    Raises:
    Exception: If any error occurs while tagging the data.
    """
    try:
        for i in range(len(entity_dict)):
            for key, values in entity_dict[i].items():
                for j in range(len(values)):
                    for x in range(len(values[j])):
                        add_tag_in_csv(df, i, key, values[j][x], x,unique_labels)
    except Exception as e:
        logs.error(f"Error tagging data: {e}")
        raise e
    return df


def clean_dataframe(df):
    """
    Cleans the text data in the DataFrame by removing whitespace and punctuation.

    Args:
    df (DataFrame): The DataFrame to clean.

    Returns:
    DataFrame: The cleaned DataFrame.

    Raises:
    Exception: If any error occurs while cleaning the DataFrame.
    """
    logs.info("Data frame cleaning")
    whitespace = string.whitespace
    punctuation = "!#%&\'()*+:,;<=>?[\\]^`{|}~-"
    table_whitespace = str.maketrans('', '', whitespace)
    table_punctuation = str.maketrans('', '', punctuation)

    def clean_text(txt):
        try:
            text = str(txt).lower()
            text = text.translate(table_whitespace)
            text = text.translate(table_punctuation)
            return text
        except Exception as e:
            logs.error(f"Error cleaning DataFrame text: {e}")
            return txt

    try:
        df['text'] = df['text'].apply(clean_text)
        df = df.query("text != '' ").dropna()
    except Exception as e:
        logs.error(f"Error cleaning DataFrame: {e}")
        raise e
    return df


def create_tagged_data(data,model_type):
    """
    Processes training data to create tagged data for model training and testing.

    Args:
    data (list): A list of dictionaries containing the training data.
    model_type (str): The type of model for which the data is being prepared.

    Raises:
    Exception: If any error occurs during the processing of tagged data.

    This function performs the following steps:
    1. Creates a unique directory for the current run based on the model type.
    2. Converts the training data into a DataFrame and saves it as an Excel file.
    3. Creates an entity dictionary and unique labels from the training data.
    4. Reads the untagged data from the Excel file and tags it using the entity dictionary.
    5. Saves the tagged data as an Excel file and a CSV file.
    6. Reads the tagged data from the CSV file and cleans it.
    7. Groups the data by 'id' and creates annotations for each group.
    8. Splits the data into training and testing sets.
    9. Saves the training and testing data as Excel files and pickles.
    """

    # Create a unique directory for this run
    paths = get_paths_instance(model_type)

    training_data = data


    all_medical_text = create_dataframe(training_data)
    save_to_excel(all_medical_text, os.path.join(paths["unique_paths"]["training_data_folder"],overall_untagged_data))

    entity_dict,unique_labels = create_entity_dict(training_data)
    df = pd.read_excel(os.path.join(paths["unique_paths"]["training_data_folder"], overall_untagged_data), engine='openpyxl')
    df = tag_data(df, entity_dict, unique_labels)
    df['tags'] = df['tags'].fillna('O')

    save_to_excel(df, os.path.join(paths["unique_paths"]["training_data_folder"], overall_tagged_data))
    df.to_csv(os.path.join(paths["unique_paths"]["training_data_folder"], data_tagged_text), sep='\t', index=False)

    try:
        with open(os.path.join(paths["unique_paths"]["training_data_folder"], data_tagged_text), mode='r', encoding='utf8', errors='ignore') as f:
            text = f.read()

        data = list(map(lambda x: x.split('\t'), text.split('\n')))
        df = pd.DataFrame(data[1:], columns=data[0])
        df = clean_dataframe(df)

        group = df.groupby(by='id')
        all_cards_data = []

        for card in group.groups.keys():
            card_data = []
            group_array = group.get_group(card)[['text', 'tags']].values
            content = ''
            annotations = {'entities': []}
            start, end = 0, 0

            for text, label in group_array:
                text = str(text)
                string_length = len(text) + 1
                start = end
                end = start + string_length

                if label != 'O':
                    annotations['entities'].append((start, end - 1, label))

                content += text + ' '

            all_cards_data.append((card,content, annotations))


        random.seed(42)
        random.shuffle(all_cards_data)
        split_idx = int(len(all_cards_data)*.8)
        train_data = all_cards_data[:split_idx]
        train_ids = [int(id) for id,_,_ in train_data]
        test_data = all_cards_data[split_idx:]
        test_ids = [int(id) for id,_,_ in test_data]
        df = pd.read_excel(os.path.join(paths['unique_paths']['training_data_folder'],overall_tagged_data), engine='openpyxl')

        train_df = df[df['id'].isin(train_ids)]
        test_df = df[df['id'].isin(test_ids)]

        save_to_excel(train_df, os.path.join(paths['unique_paths']['training_data_folder'],trf_train_data))
        save_to_excel(test_df, os.path.join(paths['unique_paths']['testing_data_folder'],trf_test_data))

        pickle.dump(train_data, open(os.path.join(paths['unique_paths']['training_data_folder'], 'TrainData.pickle'), mode='wb'))
        pickle.dump(test_data, open(os.path.join(paths['unique_paths']['testing_data_folder'], 'TestData.pickle'), mode='wb'))

    except Exception as e:
        logs.error(f"Error processing tagged data: {e}")
        raise e
    #logs.info("Running annotation done. ")