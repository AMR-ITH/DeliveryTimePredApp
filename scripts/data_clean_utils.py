import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from pathlib import Path
import dagshub
import mlflow.client


# set the output as pandas
set_config(transform_output='pandas')


columns_to_drop =  ['rider_id',
                    'restaurant_latitude',
                    'restaurant_longitude',
                    'delivery_latitude',
                    'delivery_longitude',
                    'order_date',
                    "order_time_hour",
                    "order_day",
                    "city_name",
                    "order_day_of_week",
                    "order_month"]


def change_column_names(data: pd.DataFrame):
    return (
        data.rename(str.lower,axis=1)
        .rename({
            "delivery_person_id" : "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type"},
            #"time_taken(min)": "time_taken"},
            axis=1)
    )


def data_cleaning(data: pd.DataFrame):
    minors_data = data.loc[data['age'].astype('float') < 18]
    minor_index = minors_data.index.tolist()
    six_star_data = data.loc[data['ratings'] == "6"]
    six_star_index = six_star_data.index.tolist()

    return (
        data
        .drop(columns="id")
        .drop(index=minor_index)                                                # Minor riders in data dropped
        .drop(index=six_star_index)                                             # six star rated drivers dropped
        .replace("NaN ",np.nan)                                                 # missing values in the data
        .assign(
            # city column out of rider id
            city_name = lambda x: x['rider_id'].str.split("RES").str.get(0),
            # convert age to float
            age = lambda x: x['age'].astype(float),
            # convert ratings to float
            ratings = lambda x: x['ratings'].astype(float),
            # absolute values for location based columns
            restaurant_latitude = lambda x: x['restaurant_latitude'].abs(),
            restaurant_longitude = lambda x: x['restaurant_longitude'].abs(),
            delivery_latitude = lambda x: x['delivery_latitude'].abs(),
            delivery_longitude = lambda x: x['delivery_longitude'].abs(),
            # order date to datetime and feature extraction
            order_date = lambda x: pd.to_datetime(x['order_date'],
                                                  dayfirst=True),
            order_day = lambda x: x['order_date'].dt.day,
            order_month = lambda x: x['order_date'].dt.month,
            order_day_of_week = lambda x: x['order_date'].dt.day_name().str.lower(),
            is_weekend = lambda x: (x['order_date']
                                    .dt.day_name()
                                    .isin(["Saturday","Sunday"])
                                    .astype(int)),
            # time based columns
            order_time = lambda x: pd.to_datetime(x['order_time'],
                                                   errors='coerce'),
            order_picked_time = lambda x: pd.to_datetime(x['order_picked_time'],
                                                         format='mixed'),
            # time taken to pick order
            pickup_time_minutes = lambda x: (
                                            (x['order_picked_time'] - x['order_time'])
                                            .dt.seconds / 60
                                            ),
            # hour in which order was placed
            order_time_hour = lambda x: x['order_time'].dt.hour,
            # time of the day when order was placed
            order_time_of_day = lambda x: (
                                x['order_time_hour'].pipe(time_of_day)),
            # categorical columns
            weather = lambda x: (
                                x['weather']
                                .str.replace("conditions ","")
                                .str.lower()
                                .replace("nan",np.nan)),
            traffic = lambda x: x["traffic"].str.rstrip().str.lower(),
            type_of_order = lambda x: x['type_of_order'].str.rstrip().str.lower(),
            type_of_vehicle = lambda x: x['type_of_vehicle'].str.rstrip().str.lower(),
            festival = lambda x: x['festival'].str.rstrip().str.lower(),
            city_type = lambda x: x['city_type'].str.rstrip().str.lower(),
            # multiple deliveries column
            multiple_deliveries = lambda x: x['multiple_deliveries'].astype(float))
            # target column modifications
            # time_taken = lambda x: (x['time_taken']
            #                         .str.replace("(min) ","")
            #                         .astype(int)))
        .drop(columns=["order_time","order_picked_time"])
    )
    
    
    
def clean_lat_long(data: pd.DataFrame, threshold=1):
    location_columns = ['restaurant_latitude',
                        'restaurant_longitude',
                        'delivery_latitude',
                        'delivery_longitude']

    return (
        data
        .assign(**{
            col: (
                np.where(data[col] < threshold, np.nan, data[col].values)
            )
            for col in location_columns
        })
    )
    
    
# extract day, day name, month and year
def extract_datetime_features(ser):
    date_col = pd.to_datetime(ser,dayfirst=True)

    return (
        pd.DataFrame(
            {
                "day": date_col.dt.day,
                "month": date_col.dt.month,
                "year": date_col.dt.year,
                "day_of_week": date_col.dt.day_name(),
                "is_weekend": date_col.dt.day_name().isin(["Saturday","Sunday"]).astype(int)
            }
        ))
    
    
def time_of_day(ser):

    return(
        pd.cut(ser,bins=[0,6,12,17,20,24],right=True,
               labels=["after_midnight","morning","afternoon","evening","night"])
    )


def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = data.drop(columns=columns)
    return df


def calculate_haversine_distance(df):
    location_columns = ['restaurant_latitude',
                        'restaurant_longitude',
                        'delivery_latitude',
                        'delivery_longitude']
    
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(
        dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return (
        df.assign(
            distance = distance)
    )

def create_distance_type(data: pd.DataFrame):
    return(
        data
        .assign(
                distance_type = pd.cut(data["distance"],bins=[0,5,10,15,25],
                                        right=False,labels=["short","medium","long","very_long"])
    ))


def perform_data_cleaning(data: pd.DataFrame):
    
    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
        .pipe(drop_columns,columns=columns_to_drop)
    )
    
    return cleaned_data.dropna()
    
    

if __name__ == "__main__":
    # data path for data
    
    current = Path.cwd()
    print(f'Current working directory: {current}')
    DATA_PATH = current / 'data'/'raw'/'swiggy.csv'
    print(f'Data path: {DATA_PATH}')
    
    # columns to drop
    
    # read the data from path
    df = pd.read_csv(DATA_PATH)
    print('swiggy data loaded successfuly')
    print(df.loc[0,:])  
    print("*"*50)
    # Convert the single row to a DataFrame
    # single_row_df = df.iloc[[0]]  # Using double brackets to keep it as a DataFrame
    # print(perform_data_cleaning(single_row_df))

    class DeliveryData(BaseModel):
        ID: str
        Delivery_person_ID: str
        Delivery_person_Age: str
        Delivery_person_Ratings: str
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
        multiple_deliveries: str
        Festival: str
        City: str


    # Example data
    input_data = {
        "ID": "0x4607",
        "Delivery_person_ID": "INDORES13DEL02",
        "Delivery_person_Age": "37",
        "Delivery_person_Ratings": "4.9",
        "Restaurant_latitude": 22.745049,
        "Restaurant_longitude": 75.892471,
        "Delivery_location_latitude": 22.765049,
        "Delivery_location_longitude": 75.912471,
        "Order_Date": "19-03-2022",
        "Time_Orderd": "11:30:00",
        "Time_Order_picked": "11:45:00",
        "Weatherconditions": "sunny",
        "Road_traffic_density": "High",
        "Vehicle_condition": 2,
        "Type_of_order": "Snack",
        "Type_of_vehicle": "motorcycle",
        "multiple_deliveries": "0",
        "Festival": "no",
        "City": "Urban"
    }

    # Parse data using Pydantic model
    delivery = DeliveryData(**input_data)

    # Convert to DataFrame for processing
    data_dict = delivery.dict()
    df = pd.DataFrame([data_dict])

    # Assuming perform_data_cleaning is the cleaning pipeline
    cleaned_df = perform_data_cleaning(df)
    print(cleaned_df)



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
    model_name = load_model_information(current / "run_information.json")['model_name']
    print(f"Model name: {model_name}")

    # stage of the model
    alias_name = "staging_latest"

    # Construct the model URI using the alias
    model_uri = f"models:/{model_name}@{alias_name}"
    # Load the model using the alias
    model = mlflow.pyfunc.load_model(model_uri)
    model_path = current / "models" / "model.joblib"
    # model = load_trasformer(model_path)


    # load the preprocessor
    preprocessor_path = current /"models"/"preprocessor.joblib"
    preprocessor = load_trasformer(preprocessor_path)

    # build the model pipeline
    model_pipe = Pipeline(steps=[
        ('preprocess',preprocessor),
        ("regressor",model)
    ])
    transformed_data = preprocessor.transform(cleaned_df)

    # Make prediction using the transformed data
    prediction = model.predict(transformed_data)
    print("\nPredicted delivery time:", prediction[0], "minutes")


 
