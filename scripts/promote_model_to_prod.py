import mlflow
import dagshub
import json
from pathlib import Path
from mlflow.tracking import MlflowClient


import dagshub
dagshub.init(repo_owner='AMR-ITH', repo_name='swiggy-delivery-time-estimator', mlflow=True)

# set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/AMR-ITH/swiggy-delivery-time-estimator.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

# Initialize the MLflow client
client = MlflowClient()
# Define the model name and current alias
model_name = load_model_information("run_information.json")["model_name"]
current_alias = "staging_latest"
new_alias = "production_latest"
# Get the model version associated with the current alias
model_version_info = client.get_model_version_by_alias(model_name, current_alias)
model_version = model_version_info.version
# Set the new alias for the model version
client.set_registered_model_alias(model_name, new_alias, model_version)
# Update the tag from 'staging' to 'production'
client.set_model_version_tag(
    name=model_name,
    version=model_version,
    key="deployment_stage",
    value="production"
)
print(f"Transitioned model version (model_version) from alias '{current_alias}' to '{new_alias}' and updated deployment_stage to 'production'.")