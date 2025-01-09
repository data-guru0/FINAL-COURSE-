import yaml
import os
import pandas as pd
from src.custom_exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)



##### Reading YAML files ####################

def read_yaml(file_path):

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"Successfully read YAML file: {file_path}")
            return config
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {e}")
        raise CustomException("Failed to read YAML file", e)
    
def load_data(path):
    """
    Load the dataset from the given path.
    """
    try:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise CustomException("Failed to load data", e)