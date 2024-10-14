import numpy as np
import pandas as pd
import spacy
import re
import string
import warnings
from src.config.base_config import *
from simpletransformers.ner import NERModel
from src.utilities.fileio import read_file, write_file
from src.config import logger

# Set up the logger
logs = logger.get_logger(__name__)
warnings.filterwarnings('ignore')

# Load NER model
try:
    #logs.info("Loading NER Model...")
    class NERModelSingleton:
        _instance = None
        _model = None
        _model_path= None

        def __new__(cls, model_path,reload=False):
            if cls._instance is None or reload and cls._model_path != model_path:
                cls._instance = super(NERModelSingleton, cls).__new__(cls)
                cls._model = spacy.load(model_path)
                cls._model_path = model_path
            return cls._instance

        @classmethod
        def get_model(cls):
            return cls._model
except Exception as e:
    logs.error(f"Error loading NER model: {e}")

def cleanText(txt):
    """
    Cleans the input text by removing whitespace and punctuation, and converting it to lowercase.

    Args:
    txt (str): The text to be cleaned.

    Returns:
    str: The cleaned text.

    Raises:
    Exception: If any error occurs during text cleaning.
    """
    try:
        whitespace = string.whitespace
        punctuation = "!#%&\'()*+:;,<=>?[\\]^`{|}~"
        tableWhitespace = str.maketrans('', '', whitespace)
        tablePunctuation = str.maketrans('', '', punctuation)
        text = str(txt)
        text = text.lower()
        removewhitespace = text.translate(tableWhitespace)
        removepunctuation = removewhitespace.translate(tablePunctuation)
        return str(removepunctuation)
    except Exception as e:
        logs.error(f"Error cleaning text: {e}")
        return txt

# group the label
class groupgen():
    """
    A class used to generate unique group IDs for text entries.

    Attributes:
    id (int): The current group ID.
    text (str): The last processed text.

    Methods:
    getgroup(text):
    Returns the group ID for the given text. If the text is the same as the last processed text,
    returns the current group ID. Otherwise, increments the group ID and updates the last processed text.
    """
    def __init__(self):
        self.id = 0
        self.text = ''

    def getgroup(self, text):
        try:
            if self.text == text:
                return self.id
            else:
                self.id += 1
                self.text = text
                return self.id
        except Exception as e:
            logs.error(f"Error in group generation: {e}")
            return self.id


grp_gen = groupgen()

def getPredictions_spacy(text,model_path):
    """
    Generates predictions for named entities in the input text using a specified NER model.

    Args:
    text (str): The input text to be processed.
    model_path (str): The path to the NER model.

    Returns:
    dict: A dictionary containing the predicted entities, with entity labels as keys and lists of entity values as values.

    Raises:
    Exception: If any error occurs during the prediction process.

    This function performs the following steps:
    1. Loads the NER model from the specified path.
    2. Extracts the labels from the NER model and cleans them.
    3. Splits the input text into individual words and cleans them.
    4. Converts the cleaned text into a single content string.
    5. Uses the NER model to generate predictions for the content.
    6. Converts the predictions to a JSON format and extracts tokens and labels.
    7. Merges the tokens and labels with the cleaned text DataFrame.
    8. Groups the labeled tokens and aggregates them into entities.
    9. Parses and formats the entities based on their labels.

    Example:
    >>> text = "John Doe is 45 years old and lives in New York."
    >>> model_path = "/path/to/ner/model"
    >>> getPredictions(text, model_path)
    {'PERSON': ['John Doe'], 'AGE': ['45'], 'LOCATION': ['New York']}
    """
    try:

        model_ner = NERModelSingleton(model_path,reload=True).get_model()
        ner = model_ner.get_pipe("ner")
        model_labels=ner.labels
        #print(f"labels== {model_labels}")
        cleaned_labels = [label[2:] for label in model_labels]

        # Create the dictionary with unique labels as keys and empty lists as values
        entities = {label: [] for label in set(cleaned_labels)}


        dataList = str(text).split(' ')
        dataList = [sentence for sentence in dataList if sentence.strip()]

        # Create a DataFrame with each sentence as a row
        df = pd.DataFrame(dataList, columns=['text'])
        logs.info("Text cleaning...")
        df['text'] = df['text'].apply(cleanText)

        # convert data into content
        df_clean = df.query('text != "" ')
        content = " ".join([w for w in df_clean['text']])
        # print(content)

        # get prediction from NER model
        logs.info("Getting predictions")
        doc = model_ner(content)

        # converting doc to json
        docjson = doc.to_json()
        doc_text = docjson['text']

        # creating tokens
        datafram_tokens = pd.DataFrame(docjson['tokens'])
        datafram_tokens['token'] = datafram_tokens[['start', 'end']].apply(
        lambda x: doc_text[x[0]:x[1]], axis=1)

        right_table = pd.DataFrame(docjson['ents'])[['start', 'label']]
        datafram_tokens = pd.merge(datafram_tokens, right_table, how='left', on='start')
        datafram_tokens.fillna('O', inplace=True)

        # join label to df_clean dataframe
        df_clean['end'] = df_clean['text'].apply(lambda x: len(x) + 1).cumsum() - 1
        df_clean['start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis=1)

        # inner join with start
        dataframe_info = pd.merge(df_clean, datafram_tokens[['start', 'token', 'label']], how='inner', on='start')

        bb_df = dataframe_info.query("label != 'O' ")

        logs.info("group generating...")
        bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])
        bb_df['group'] = bb_df['label'].apply(grp_gen.getgroup)

        col_group = ['label', 'text', 'group']
        group_tag_img = bb_df[col_group].groupby(by='group')
        img_tagging = group_tag_img.agg({
        'label': np.unique,
        'text': lambda x: " ".join(x)
        })

        # Entities
        info_arr = dataframe_info[['text', 'label']].values
        
        previous = 'O'
        #logs.info("Parsing text")
        for token, label in info_arr:
            bio_tag = label[:1]
            label_tag = label[2:]
            # step 1 parse the token
            # text = parser(token, label_tag)
            text=token

            if bio_tag in ('B', "I"):
                if previous != label_tag:
                    #print(label_tag)
                    entities[label_tag].append(text)
                else:
                    if bio_tag == 'B':
                        entities[label_tag].append(text)
                    else:
                        if label_tag in entities:
                            entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                        else:
                            entities[label_tag][-1] = entities[label_tag][-1] + text
            previous = label_tag

        return entities
    except Exception as e:
        logs.error(f"Error in getPredictions from spacy: {e}")
        return {}


class TransformerPredict:

    """
    A class that handles the prediction of labels for given input text using a transformer-based NER model.

    Attributes:
    NERModel (NERModel): An instance of the NERModel class for named entity recognition.

    Methods:
    clean_text(text):
    Cleans the input text by removing unwanted characters and formatting it.

    label_generation(model_outputs):
    Generates labeled entities from the model outputs.

    get_prediction(text):
    Predicts the labels for the given input text.
    """

    def __init__(self,output_path = None):
        """
        Initializes the TransformerPredict class with the specified output path.

        Args:
        output_path (str, optional): The path to save the model outputs. Defaults to None.
        """

        self.NERModel = NERModel(TRF_MODEL_TYPE,output_path,use_cuda=False)

    def clean_text(self,text):
        """
        Cleans the input text by removing unwanted characters and formatting it.

        Args:
        text (str): The text to be cleaned.

        Returns:
        list: A list of cleaned and split text segments.

        Raises:
        Exception: If any error occurs during text cleaning.
        """
        logs.info("Text cleaning...")
        try:
            text = text.lower()
            text = re.sub(r"[$]([\d]+[,][\d]+)+", lambda match: match.group().replace(",",""), text)
            text = text.replace("-"," ")
            text = text.replace('\n', ' ')
            text = text.replace('(', '')
            text = text.replace(')', '')
            text = re.sub(r",", " ", text)
            text = re.sub(r"\.", " ", text)
            dataList = re.split("-| ", str(text))
            dataList = [sentence for sentence in dataList if sentence.strip()]
            return dataList

        except Exception as e:
            logs.error(f"Error cleaning text: {e}")
            raise

    def label_generation(self,model_outputs):
        """
        Generates labeled entities from the model outputs.

        Args:
        model_outputs (list): The outputs from the NER model.

        Returns:
        list: A list of dictionaries containing labeled entities.

        Raises:
        Exception: If any error occurs during label generation.
        """
        try:
            output = []
            for example in model_outputs:
                entities = {}
                current_entity = None
                current_value = []
                for item in example:
                    for key, value in item.items():
                        if value.startswith('B-'):
                            if current_entity:
                                if current_entity not in entities:
                                    entities[current_entity] = []
                                entities[current_entity].append(' '.join(current_value))
                            current_entity = value[2:]
                            current_value = [key]
                        elif value.startswith('I-'):
                            if not current_entity or value[2:] != current_entity:
                                if current_entity:
                                    if current_entity not in entities:
                                        entities[current_entity] = []
                                    entities[current_entity].append(' '.join(current_value))
                                current_entity = value[2:]
                                current_value = []
                            current_value.append(key)
                        elif value == 'O':
                            if current_entity:
                                if current_entity not in entities:
                                    entities[current_entity] = []
                                entities[current_entity].append(' '.join(current_value))
                                current_entity = None
                                current_value = []

                # Handle the last entity if exists
                if current_entity:
                    if current_entity not in entities:
                        entities[current_entity] = []
                    entities[current_entity].append(' '.join(current_value))
                output.append(entities)
            return output

        except Exception as e:
            logs.error(f"Error in label generation: {e}")
            raise

    def get_prediction(self,text):
        """
        Predicts the labels for the given input text.

        Args:
        text (str): The input text to be labeled.

        Returns:
        dict: A dictionary containing the predicted entities and their labels.

        Raises:
        Exception: If any error occurs during prediction.
        """
        try:
            text = text.lower()
            clean_data = self.clean_text(text)
            clean_data = " ".join(clean_data)

            model_pred,_ = self.NERModel.predict([clean_data])
            #print(model_pred)
            final_prediction = self.label_generation(model_pred)

            return final_prediction[0]
        except Exception as e:
            logs.error(f"Error in getPredictions: {e}")
            raise