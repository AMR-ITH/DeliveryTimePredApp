import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import set_config
import json
import sys

import dagshub
dagshub.init(repo_owner='AMR-ITH', repo_name='swiggy-delivery-time-estimator', mlflow=True)

# Add the src directory to the Python path to import the logging config
src_dir = str(Path(__file__).parent.parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.logging_config import setup_logging

# Set up logging
logger = setup_logging("data_evaluation")

# set the transformer outputs to pandas
set_config(transform_output='pandas')

# set the mlflow tracking server URI
mlflow.set_tracking_uri("https://dagshub.com/AMR-ITH/swiggy-delivery-time-estimator.mlflow")

# set mlflow experiment name 
mlflow.set_experiment("DVC-Pipeline")

target_col = "time_taken"

# Load the dataset
def load_data(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")

# load the model
def load_model(model_path: Path):
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")

def save_model_info(save_json_path, run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path, 'w') as f:
        json.dump(info_dict, f, indent=4)
    logger.info(f"Model info saved to {save_json_path}")

if __name__ == "__main__":

    # Define the root path
    root_path = Path(__file__).parent.parent.parent
    logger.info(f"Root path: {root_path}")

    # train data load path 
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"

    # model path
    model_path = root_path / "models" / "model.joblib"

    # Load the training data
    train_df = load_data(train_data_path)
    test_df = load_data(test_data_path)

    # split the train and test data 
    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    # load the model
    model = load_model(model_path)



    # get the train and test predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # calculate the train and test mae
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    logger.info(f"Train MAE: {train_mae}, Test MAE: {test_mae}")

    # calculate the r2 scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    logger.info(f"Train R2: {train_r2}, Test R2: {test_r2}")

    # calculate cross val scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    logger.info(f"Cross-validation scores: {cv_scores}")

    # mean cross val error
    mean_cv_error = -(cv_scores.mean())
    logger.info(f"Mean cross-validation error: {mean_cv_error}")

    with mlflow.start_run() as run:
        # set tags
        mlflow.set_tag("model", "Food Delivery Time Estimator")

        # log parameters
        mlflow.log_params(model.get_params())

        # log metrics
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("cv_score", mean_cv_error)
        mlflow.log_metrics({f"CV_{num}": -score for num, score in enumerate(cv_scores)})

        # mlflow dataset input datatype
        train_data_input = mlflow.data.from_pandas(train_df, targets=target_col)
        test_data_input = mlflow.data.from_pandas(test_df, targets=target_col)

        # log datasets
        mlflow.log_input(context="training", dataset=train_data_input)
        mlflow.log_input(context="validation", dataset=test_data_input)

        # model signature
        model_signature = mlflow.models.infer_signature(
            model_input=X_train.sample(20, random_state=42),
            model_output=model.predict(X_train.sample(20, random_state=42))
        )

        # log the final model
        mlflow.sklearn.log_model(model, "delivery_time_pred_model", signature=model_signature)

        # log stacking regressor
        mlflow.log_artifact(root_path / "models" / "stacking_model.joblib")

        # log the power transformer
        mlflow.log_artifact(root_path / "models" / "power_transformer.joblib")

        # log the preprocessor
        mlflow.log_artifact(root_path / "models" / "preprocessor.joblib")

        # get the current run artifact uri
        artifact_uri = mlflow.get_artifact_uri()

    # get the run id
    run_id = run.info.run_id
    model_name = "delivery_time_pred_model"

    # save the model info
    save_json_path = root_path / "run_information.json"
    save_model_info(save_json_path=save_json_path,
                    run_id=run_id,
                    artifact_path=artifact_uri,
                    model_name=model_name)

