import pytest
import mlflow
import json

import dagshub
dagshub.init(repo_owner='AMR-ITH', repo_name='swiggy-delivery-time-estimator', mlflow=True)

# set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/AMR-ITH/swiggy-delivery-time-estimator.mlflow")

# Helper function to load model name from file
def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

@pytest.mark.parametrize("alias_name", ["staging_latest"])
def test_load_model_using_alias(alias_name):
    model_info = load_model_information("run_information.json")
    model_name = model_info["model_name"]

    # Construct the model URI using the alias
    model_uri = f"models:/{model_name}@{alias_name}"

    # Try loading the model
    model = mlflow.pyfunc.load_model(model_uri)

    # Assert model was loaded successfully
    assert model is not None, f"Failed to load model from alias '{alias_name}'"

    print(f"Model '{model_name}' loaded successfully from alias '{alias_name}'")
