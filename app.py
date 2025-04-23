from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from scripts.data_clean_utils import perform_data_cleaning
from pathlib import Path

# set the output as pandas
set_config(transform_output='pandas')

# initialize dagshub
import dagshub
import mlflow.client

# Initialize FastAPI app
dagshub.init(repo_owner='AMR-ITH', repo_name='swiggy-delivery-time-estimator', mlflow=True)

class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: int
    Delivery_person_Ratings: float
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: int
    Festival: str
    City: str

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

def load_trasformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

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




# load the model info to get the model name
model_name = load_model_information("run_information.json")['model_name']
print(f"Model name: {model_name}")

# stage of the model
alias_name = "staging_latest"

# Construct the model URI using the alias
model_uri = f"models:/{model_name}@{alias_name}"
# Load the model using the alias
model = mlflow.pyfunc.load_model(model_uri)

# load the preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_trasformer(preprocessor_path)



# create the app
app = FastAPI()

# create the home endpoint
@app.get(path="/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

# create the predict endpoint
@app.post(path="/predict")
def do_predictions(data: Data):
    pred_data = pd.DataFrame({
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition,
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City
        },index=[0]
    )
    
    print("Input data shape:", pred_data.shape)
    print("Input data columns:", pred_data.columns.tolist())
    
    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data)
    
    print("Cleaned data shape:", cleaned_data.shape)
    print("Cleaned data columns:", cleaned_data.columns.tolist())

    # Extract transformed data using only the preprocessor
    transformed_df = preprocessor.transform(cleaned_data)

    # Convert the transformed data to a DataFrame
    transformed_df = pd.DataFrame(transformed_df, columns=preprocessor.get_feature_names_out())

    # Print the data types of the transformed DataFrame
    print("Data types of transformed DataFrame:")
    print(transformed_df.dtypes)
    print("*" * 60)
    # Ensure the 'vehicle_condition' column is of type int64
    if 'vehicle_condition' in transformed_df.columns:
        transformed_df['vehicle_condition'] = transformed_df['vehicle_condition'].astype('int64')
    print("#" * 60)

    # Print the data types of the transformed DataFrame
    print("Data types of transformed DataFrame:")
    print(transformed_df.dtypes)

    print("*" * 60)
    print("Transformed data shape:", transformed_df.shape)
    print("Transformed data columns:", transformed_df.columns.tolist())
    print("*" * 60)
    print("Transformed data first few rows:", transformed_df.head())

    # Then predict
    prediction = model.predict(transformed_df)
    return {"predicted_delivery_time": round(float(prediction[0]), 2)}


   
   
if __name__ == "__main__":
    uvicorn.run(app="app:app",host="0.0.0.0",port=8000)





