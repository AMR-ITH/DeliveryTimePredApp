import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path
import numpy as np
import sys

# Add the src directory to the Python path to import the logging config
src_dir = str(Path(__file__).parent.parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.logging_config import setup_logging

# Set up logging
logger = setup_logging("data_preparation")

def load_data(cleaned_data_path: Path) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from {cleaned_data_path}")
        df = pd.read_csv(cleaned_data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {cleaned_data_path}: {e}")

def preprocess_data_parameters(params_path: Path) -> dict:
    try:
        logger.info(f"Loading preprocessing parameters from {params_path}")
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.info("Preprocessing parameters loaded successfully")
        return params
    except Exception as e:
        logger.error(f"Error loading preprocessing parameters from {params_path}: {e}")

def test_train_split_save(df: pd.DataFrame, params: dict, 
                          preprocessed_data_path_train: Path, preprocessed_data_path_test: Path):
    try:
        test_size = params["Data_Preparation"]["test_size"]
        random_state = params["Data_Preparation"]["random_state"]

        logger.info("Performing train-test split")
        X_train, X_test = train_test_split(df, test_size=test_size, random_state=random_state)
        logger.info(f"Train-test split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # save preprocessed data
        logger.info(f"Saving preprocessed train data to {preprocessed_data_path_train}")
        X_train.to_csv(preprocessed_data_path_train, index=False)
        logger.info(f"Saving preprocessed test data to {preprocessed_data_path_test}")
        X_test.to_csv(preprocessed_data_path_test, index=False)
        logger.info("Preprocessed data saved successfully")
    except Exception as e:
        logger.error(f"Error during train-test split and save: {e}")

if __name__ == "__main__":
    try:
        logger.info("Starting data preparation script")

        # root path
        root_path = Path(__file__).parent.parent.parent
        logger.info(f"Root path: {root_path}")

        # cleaned data path
        cleaned_data_path = root_path / "data" / "cleaned" / "swiggy_cleaned.csv"
        # load the cleaned data
        df = load_data(cleaned_data_path)

        # load preprocessing parameters
        params_path = root_path / "params.yaml"
        params = preprocess_data_parameters(params_path)

        # path to save preprocessed data
        save_data_dir = root_path / "data" / "interim"
        save_data_dir.mkdir(exist_ok=True, parents=True)
        preprocessed_data_path_train = save_data_dir / "train.csv"
        preprocessed_data_path_test = save_data_dir / "test.csv"

        # save preprocessed data
        test_train_split_save(df, params, preprocessed_data_path_train, preprocessed_data_path_test)
        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"An error occurred in the data preparation script: {e}")


    

    
