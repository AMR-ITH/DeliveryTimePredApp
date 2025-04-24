import pytest
import mlflow
import dagshub
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from sklearn import set_config

# set the output as pandas
set_config(transform_output='pandas')



import dagshub
dagshub.init(repo_owner='AMR-ITH', repo_name='swiggy-delivery-time-estimator', mlflow=True)

# set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/AMR-ITH/swiggy-delivery-time-estimator.mlflow")





def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

def load_trasformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer
# set the root path
root_path = Path(__file__).parent.parent

# load the preprocessor
preprocessor_path = root_path / "models" / "preprocessor.joblib"

# load the preprocessor
preprocessor = load_trasformer( preprocessor_path)


# Try loading the model
model_name = load_model_information( "run_information.json")["model_name"]
alias_name = "staging_latest"
# Construct the model URI using the alias
model_uri = f"models:/{model_name}@{alias_name}"
model = mlflow.pyfunc.load_model(model_uri)


# test data path
test_data_path = root_path / "data" / "interim" / "test.csv"




@pytest.mark.parametrize(argnames="model, test_data_path, threshold_error",
                        argvalues=[(model, test_data_path, 5)])
def test_model_performance(model,test_data_path,threshold_error):
    # load test data
    df = pd.read_csv(test_data_path)
    
    # drop the missing values
    df.dropna(inplace=True)
    
    # make X and y
    X = df.drop(columns=["time_taken"])
    y = df['time_taken'].str.replace(r'\(min\)\s*', '', regex=True).astype(float)

    # transform the data
    X_trans = preprocessor.transform(X)

    # make predictions
    y_pred = model.predict(X_trans)

    # calculate the error
    error = mean_absolute_error(y,y_pred)

    # assert the error is below the threshold
    assert error < threshold_error, f"Model performance is below the threshold. Error: {error}"
    print(f"Model performance test passed. Error: {error}")
    print(f"The {model_name} model passed the performance test")


