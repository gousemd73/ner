import os
from src.config.base_config import trf_train_data, TRF_MODEL_NAME,TRF_MODEL_TYPE,trf_test_data
import pandas as pd
from simpletransformers.ner import NERModel,NERArgs
from seqeval.metrics import accuracy_score,classification_report
import warnings
import torch
from src.config import logger


# Set up the logger
logs = logger.get_logger(__name__)
warnings.filterwarnings("ignore")

# cfg_paths = get_paths_instance()
# cfg_paths = cfg_paths['unique_paths']

def read_data(file_path):

    '''Method that reads processed excel file that
    was created after text processing'''
    #logs.info("Reading data from preprocessed text")
    try:
        data = pd.read_excel(file_path)
        data = data.rename(columns={"id":"sentence_id","text":"words","tags":"labels"})
        labels = data.labels.unique()
        labels = list(labels)
        return data, labels
    except Exception as e:
        logs.error(f"Error in reading file path: {e}")
        raise e


def NerModel(model_type,model_name,labels,model_output_folder):
    '''Defining model arguments and model specification.....
    '''
    logs.info("Defining Model arguments and model specification.....")
    try:

        # Get the current date and time

        args = NERArgs()
        args.output_dir = os.path.join(model_output_folder,'bert_model/')
        args.best_model_dir = os.path.join(model_output_folder,'bert_model/best_model/')


        args.no_cache = True
        args.evaluate_during_training_steps = 200
        args.learning_rate = 4e-6
        args.num_train_epochs = 5
        args.train_batch_size = 16
        args.eval_batch_size = 8
        args.save_steps = -1
        args.max_seq_length = 512

        args.evaluate_during_training = True
        args.save_eval_checkpoints = False
        args.overwrite_output_dir = True
        args.save_eval_checkpoints =False
        args.save_model_every_epoch = False
        args.save_best_model = True
        args.save_optimizer_and_scheduler = False
        args.classification_report = True
        device = True if torch.cuda.is_available() else False
        args.use_multiprocessing = False
        args.use_multiprocessing_for_evaluation = False
        model = NERModel(model_type, model_name,labels=labels,use_cuda=device,args=args)

        return model
    except Exception as e:
        logs.error(f"Error in model initialization: {e}")
        raise e

def train_model(model,train_dataframe,eval_dataframe):
    '''Method that starts training the model for arguments
    that were set during model initiation.....
    '''
    logs.info("Model training initiated...")
    try:

        global_step, training_details = model.train_model(train_data=train_dataframe,eval_data=eval_dataframe,accuracy=accuracy_score)

        return training_details

    except Exception as e:
        logs.error(f"Error in training model: {e}")
        raise e

def eval_model(model,eval_dataframe,output):

    '''Method for evaluation trained model on evaluation dataset...'''
    logs.info("Evaluating Model...")
    try:
        result, model_outputs, preds_list = model.eval_model(eval_dataframe,output_dir = output,accuracy=accuracy_score)

        return pd.DataFrame.from_dict([result])

    except Exception as e:
        logs.error(f"Error in evaluation of model: {e}")
        raise e


class Model:
    def __init__(self,unique_paths):
        self.unique_paths = unique_paths

    def get_best_model_results(self,path):
        # logs.info("Choosing best model available...")
        # try:
        #     if os.path.exists(path):
        #         with open(os.path.join(path,'eval_results.txt'),'r') as f:
        #             data = f.readlines()
        #         metrics_dict = {}
        #         for metric in data:
        #             key, value = metric.split('=')
        #             metrics_dict[key.strip()] = float(value.strip())
        #         return metrics_dict
        # except Exception as e:
        #     logs.error(f"Error in getting best model metrics: {e}")
        #     raise e
        metrics = {}
        with open(os.path.join(path,'eval_results.txt'), 'r') as file:
            lines = file.readlines()
            
            # Parsing individual classes and their metrics
            i = 2  # Skip the first two lines (header lines)
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith("micro avg") or line.startswith("macro avg") or line.startswith("weighted avg"):
                    avg_type,avg, precision, recall, f1_score, support = line.split()
                    metrics[avg_type+avg] = {
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1-score": float(f1_score),
                        "support": int(support)
                    }
                elif '=' in line:
                    # Handle accuracy, eval_loss, etc.
                    key, value = line.split('=', 1)
                    metrics[key.strip()] = float(value.strip())
                else:
                    # Handle other class metrics
                    if not line:
                        i += 1
                        continue  # Skip empty lines
                    parts = line.split()
                    class_name = parts[0]
                    precision, recall, f1_score, support = float(parts[1]),float(parts[2]),float(parts[3]),int(parts[4])
                    metrics[class_name] = {
                        "precision": precision,
                        "recall": recall,
                        "f1-score": f1_score,
                        "support": support
                    }
                i += 1

            return metrics



    def ner_training(self,model_type=TRF_MODEL_TYPE,model_name=TRF_MODEL_NAME):

        '''Driving method that start model traning processe for NER training using simple-transformers....
        We need to pass path to input data file along with model type and model name that we are interested to train'''
        logs.info(" Driving Code for training...")
        try:
            logs.info("Reading data from preprocessed text for training...")
            train_df, labels = read_data(os.path.join(self.unique_paths['training_data_folder'],trf_train_data))

            logs.info("Reading data from preprocessed text for evaluation...")
            eval_df, _ = read_data(os.path.join(self.unique_paths['testing_data_folder'],trf_test_data))

            model = NerModel(model_type,model_name,labels,self.unique_paths['model_output_folder'])
            training_progress = train_model(model,train_df,eval_df)
            logs.info("Saving Model path")
            saved_model_path = model.args.best_model_dir
            bm = NERModel(model_type,saved_model_path,args={"classification_report":True})
            p_df = eval_model(bm,eval_df,saved_model_path)
            best_model_metrics = self.get_best_model_results(saved_model_path)

            return [saved_model_path,best_model_metrics ]

        except Exception as e:
            logs.error(f"Error in main NER function : {e}")
            raise e


# if __name__ == "__main__":
# ner_training(model_type = TRF_MODEL_TYPE,model_name=TRF_MODEL_NAME,training_file_path=f'{config_paths.data}/transformer/Train_tagged.xlsx')
