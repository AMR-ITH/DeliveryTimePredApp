import json
import dagshub
import joblib
import mlflow
import pandas as pd

# Create a dictionary with column names and sample values
data = {
    "age": [25.0],
    "ratings": [4.5],
    "pickup_time_minutes": [15.0],
    "distance": [5.2],
    "weather_fog": [0.0],
    "weather_sandstorms": [0.0],
    "weather_stormy": [0.0],
    "weather_sunny": [1.0],
    "weather_windy": [0.0],
    "type_of_order_drinks": [0.0],
    "type_of_order_meal": [1.0],
    "type_of_order_snack": [0.0],
    "type_of_vehicle_motorcycle": [1.0],
    "type_of_vehicle_scooter": [0.0],
    "festival_yes": [0.0],
    "city_type_semi-urban": [0.0],
    "city_type_urban": [1.0],
    "is_weekend_1": [1.0],
    "order_time_of_day_evening": [0.0],
    "order_time_of_day_morning": [1.0],
    "order_time_of_day_night": [0.0],
    "traffic": [2.0],
    "distance_type": [1.0],
    "vehicle_condition": [4],  # long type, use int in pandas
    "multiple_deliveries": [0.0]
}

# Create the DataFrame
df = pd.DataFrame(data)

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

def load_trasformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

# Initialize FastAPI app
dagshub.init(repo_owner='AMR-ITH', repo_name='swiggy-delivery-time-estimator', mlflow=True)

# load the model info to get the model name
model_name = load_model_information("run_information.json")['model_name']
print(f"Model name: {model_name}")

# stage of the model
alias_name = "staging_latest"

# Construct the model URI using the alias
model_uri = f"models:/{model_name}@{alias_name}"

# Load the model using the alias (only once)
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Display the DataFrame
print(df)

# Predict on a Pandas DataFrame
prediction = loaded_model.predict(df)
print("Prediction:", prediction)