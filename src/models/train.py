import pandas as pd
import yaml
import joblib
import logging
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path
from sklearn.ensemble import StackingRegressor
import sys




TARGET = "time_taken"

# Add the src directory to the Python path to import the logging config
src_dir = str(Path(__file__).parent.parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.logging_config import setup_logging

# Set up logging
logger = setup_logging("data_training")


def load_params(params_path: Path) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f"Parameters loaded successfully from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Failed to load parameters from {params_path}: {e}")
        return {}

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        return pd.DataFrame()

def train_model(model, X_train: pd.DataFrame, y_train):
    try:
        # fit on the data
        model.fit(X_train, y_train)
        logger.info("Model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        return None

def save_model(model: object, save_dir: Path, model_name: str):
    try:
        file_path = save_dir / model_name
        joblib.dump(model, file_path)
        logger.info(f"Model {model_name} saved successfully to {save_dir}")
    except Exception as e:
        logger.error(f"Failed to save model {model_name} to {save_dir}: {e}")


if __name__ == "__main__":

    # root path
    root_path = Path(__file__).parent.parent.parent
    logger.info(f"Root path: {root_path}")

    # load parameters
    params_path = root_path / "params.yaml"
    params = load_params(params_path)

    # load the preprocessed data
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"

    train_df = load_data(train_data_path)

    # split the data into X and y
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    logger.info("Data split into features and target")

    # transform y_train jhon's method for target variable
    pt = PowerTransformer()
    y_train_pt = pt.fit_transform(y_train.values.reshape(-1, 1))
    logger.info("Target variable transformed using PowerTransformer")

    # build the best models
    rf_params = params["Train"]["Random_Forest"]
    lgbm = params["Train"]["LightGBM"]

    best_rf = RandomForestRegressor(**rf_params)
    best_lgbm = LGBMRegressor(**lgbm)
    lr = LinearRegression()

    # build the stacking regressor

    stacking_reg = StackingRegressor(estimators=[("rf", best_rf),
                                                 ("lgbm", best_lgbm)],
                                     final_estimator=lr,
                                     cv=5, n_jobs=-1)
    logger.info("Stacking regressor built")

    # build transformed regressor

    model = TransformedTargetRegressor(regressor=stacking_reg,
                                       transformer=pt)
    logger.info("TransformedTargetRegressor created")

    # fit the model on training data
    train_model(model, X_train, y_train)

    # model name
    model_filename = "model.joblib"
    # directory to save model
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)

    # extract the model from wrapper
    stacking_model = model.regressor_
    transformer = model.transformer_

    # save the model
    save_model(model=model,
               save_dir=model_save_dir,
               model_name=model_filename)

    # save the stacking model

    stacking_filename = "stacking_model.joblib"

    save_model(model=stacking_model,
               save_dir=model_save_dir,
               model_name=stacking_filename)

    # save the transformer
    transformer_filename = "power_transformer.joblib"
    save_model(model=transformer,
               save_dir=model_save_dir,
               model_name=transformer_filename)


    


