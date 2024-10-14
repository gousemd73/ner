"""
base_config.py
provides global configs
"""

import os
import datetime
from dataclasses import dataclass, field

@dataclass
class Paths:
    """
    contains paths of common directories
    """

    root_path: str = os.getcwd()
    src_path: str = os.path.join(root_path, 'src')
    output: str = os.path.join(root_path, 'output')
    models: str = os.path.join(root_path, 'models')

    def create_unique_run_folder(self,model_type):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_run_folder = os.path.join(self.models, f"{model_type}_run_{timestamp}")
        os.makedirs(unique_run_folder, exist_ok=True)
        return unique_run_folder

    def get_unique_paths(self,model_type):

        unique_run_folder = self.create_unique_run_folder(model_type)

        training_data_folder = os.path.join(unique_run_folder, "training_data")
        testing_data_folder = os.path.join(unique_run_folder, "testing_data")
        model_output_folder = os.path.join(unique_run_folder, "model_output")

        os.makedirs(training_data_folder, exist_ok=True)
        os.makedirs(testing_data_folder, exist_ok=True)
        os.makedirs(model_output_folder, exist_ok=True)

        return {
        "unique_run_folder":unique_run_folder,
        "training_data_folder": training_data_folder,
        "testing_data_folder": testing_data_folder,
        "model_output_folder": model_output_folder
        }

    def test_paths(self):
        print(f'root_path = {self.root_path}')
        print(f'src_path = {self.src_path}')
        print(f'data_path = {self.data}')
        print(f'output_path = {self.output}')
        print(f'models_path = {self.models}')

# _paths_instance = None

def get_paths_instance(model_type):
    global _paths_instance
    # if _paths_instance is None:
    config_paths = Paths()
    _paths_instance = {
    "common_paths": config_paths,
    "unique_paths": config_paths.get_unique_paths(model_type)
    }
    return _paths_instance



TRF_MODEL_NAME = "bert-base-uncased"
TRF_MODEL_TYPE = 'bert'

overall_untagged_data = "BIO-untagged_data.xlsx"
overall_tagged_data = "BIO-tagged-data.xlsx"
data_tagged_text = "BIO-tagged-data.txt"
trf_train_data = "trf_train_data.xlsx"
trf_test_data = "trf_test_data.xlsx"
dummy_data = "Train_data.json"