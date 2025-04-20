import pandas as pd
import logging
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    OrdinalEncoder)
import joblib
from sklearn import set_config
import sys

# Add the src directory to the Python path to import the logging config
src_dir = str(Path(__file__).parent.parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.logging_config import setup_logging

# Set up logging
logger = setup_logging("data_preparation")

# set the transformer outputs to pandas
set_config(transform_output='pandas')

# columns to preprocess in data

num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]

nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

target_col = "time_taken"

# generate order for ordinal encoding

traffic_order = ["low","medium","high","jam"]

distance_type_order = ["short","medium","long","very_long"]

# load the data
def load_data(file_path: Path) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")

def drop_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Dropping missing values")
        df_dropped = data.dropna()
        missing_vals = df_dropped.isna().sum().sum()
        
        if missing_vals > 0:
            logger.error("The dataframe has missing values")
            return None
        
        logger.info(f"Data shape after dropping missing values: {df_dropped.shape}")
        return df_dropped
    except Exception as e:
        logger.error(f"Error in dropping missing values: {e}")

def train_preprocessor(preprocessor: ColumnTransformer,data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Training preprocessor")
        preprocessor.fit(data)
        logger.info("Preprocessor trained successfully")
        return preprocessor
    except Exception as e:
        logger.error(f"Failed to train preprocessor: {e}")

def perform_transforamtions(preprocessor: ColumnTransformer, data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Performing transformations")
        df_trans = preprocessor.transform(data)
        logger.info("Transformations completed")
        return df_trans
    except Exception as e:
        logger.error(f"Failed to perform transformations: {e}")

def save_transformer(transformer: ColumnTransformer, save_dir: Path, transformer_name: str):
    try:
        file_path = save_dir / transformer_name
        logger.info(f"Saving transformer to {file_path}")
        joblib.dump(transformer, file_path)
        logger.info("Transformer saved successfully")
    except Exception as e:
        logger.error(f"Failed to save transformer: {e}")

if __name__ == "__main__":
    try:
        # root path
        root_path = Path(__file__).parent.parent.parent
        logger.info(f"Root path: {root_path}")

        # data path for training data and test data
        data_path = root_path / "data" / "interim"
        train_data_path = data_path / "train.csv"
        test_data_path = data_path / "test.csv"

        # load data
        train_df = load_data(train_data_path)
        test_df = load_data(test_data_path)

        # rm the missing value in train_df & test_df
        train_df = drop_missing_values(train_df)
        test_df = drop_missing_values(test_df)

        # preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("scale", MinMaxScaler(), num_cols),
                ("nominal_encode", OneHotEncoder(drop="first", handle_unknown="ignore",
                                                 sparse_output=False), nominal_cat_cols),
                ("ordinal_encode", OrdinalEncoder(categories=[traffic_order, distance_type_order],
                                                  encoded_missing_value=-999,
                                                  handle_unknown="use_encoded_value",
                                                  unknown_value=-1),
                 ordinal_cat_cols)
            ],
            remainder="passthrough", n_jobs=-1,
            force_int_remainder_cols=False, verbose_feature_names_out=False
        )

        # split the train and test data 
        X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
        X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

        # fit the preprocessor on X_train
        train_preprocessor(preprocessor=preprocessor, data=X_train)

        # transform the data
        X_train_trans = perform_transforamtions(preprocessor=preprocessor, data=X_train)
        X_test_trans = perform_transforamtions(preprocessor=preprocessor, data=X_test)

        # join X and y
        df_train_trans = pd.concat([X_train_trans, y_train], axis=1)
        df_test_trans = pd.concat([X_test_trans, y_test], axis=1)

        # save the preprocessed data
        saved_data_path = root_path / "data" / "processed"
        saved_data_path.mkdir(exist_ok=True, parents=True)
        df_train_trans.to_csv(saved_data_path / "train_trans.csv", index=False)
        df_test_trans.to_csv(saved_data_path / "test_trans.csv", index=False)
        logger.info("Preprocessed data saved successfully")

        # save the preprocessor to location
        transformer_filename = "preprocessor.joblib"
        transformer_save_dir = root_path / "models"
        transformer_save_dir.mkdir(exist_ok=True)
        save_transformer(transformer=preprocessor,
                         save_dir=transformer_save_dir,
                         transformer_name=transformer_filename)
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")

   







      


