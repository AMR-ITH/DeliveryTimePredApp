import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging
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



# set the mlflow tracking server URI
mlflow.set_tracking_uri("https://dagshub.com/AMR-ITH/swiggy-delivery-time-estimator.mlflow")

def load_model_info(file_path):
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    return model_info

if __name__ == "__main__":

    # root path
    root_path = Path(__file__).parent.parent.parent

    # run information file path
    run_info_path = root_path / "run_information.json"

    #load the info model file
    model_info = load_model_info(run_info_path)

    # get the run id and model name
    run_id = model_info["run_id"]
    model_name = model_info["model_name"]

    # model to register path
    model_registry_path = f"runs:/{run_id}/{model_name}"

    # register the model
    model_version = mlflow.register_model(model_uri=model_registry_path, name=model_name)
    logger.info(f"Model registered with URI: {model_registry_path}, Name: {model_name}")

    # get the model version
    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"Registered model version: {registered_model_version}, name: {registered_model_name}")

    # update the stage of the model to staging
    client = MlflowClient()

    # Set a tag on the model version
    client.set_model_version_tag(
        name=registered_model_name,
        version=registered_model_version,
        key="deployment_stage",
        value="staging"
    )
    # Set an alias for the model version
    alias_name = "staging_latest"
    client.set_registered_model_alias(registered_model_name, alias_name, registered_model_version)


    logger.info(f"Set model version tag: deployment_stage=staging for model {registered_model_name} version {registered_model_version}")

    logger.info("Model registration and tagging completed successfully")


     