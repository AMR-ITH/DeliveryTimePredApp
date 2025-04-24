import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging
import sys

# Add the src directory to the Python path to import the logging config
src_dir = str(Path(__file__).parent.parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.logging_config import setup_logging

# Set up logging
logger = setup_logging("data_cleaning")

# cols to drop based on EDA analysis 
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

# load data
def load_data(file_path: Path) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")

def change_column_names(data: pd.DataFrame):
    try:
        logger.info("Renaming columns")
        renamed_data = (
            data.rename(str.lower, axis=1)
            .rename({
                "delivery_person_id": "rider_id",
                "delivery_person_age": "age",
                "delivery_person_ratings": "ratings",
                "delivery_location_latitude": "delivery_latitude",
                "delivery_location_longitude": "delivery_longitude",
                "time_orderd": "order_time",
                "time_order_picked": "order_picked_time",
                "weatherconditions": "weather",
                "road_traffic_density": "traffic",
                "city": "city_type",
                "time_taken(min)": "time_taken"}, axis=1)
        )
        logger.info("Column renaming completed")
        return renamed_data
    except Exception as e:
        logger.error(f"Error renaming columns: {e}")

def time_of_day(ser: pd.Series):
    try:
        logger.debug("Converting time to time of day categories")
        return pd.cut(ser, bins=[0, 6, 12, 17, 20, 24], right=True,
                      labels=["after_midnight", "morning", "afternoon", "evening", "night"])
    except Exception as e:
        logger.error(f"Error converting time to time of day categories: {e}")

def data_cleaning(data: pd.DataFrame):
    try:
        minors_data = data.loc[data['age'].astype('float') < 18]
        minor_index = minors_data.index.tolist()
        six_star_data = data.loc[data['ratings'] == "6"]
        six_star_index = six_star_data.index.tolist()

        cleaned_data = (
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
                                                  dayfirst=True, errors='coerce'),
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

        logger.info(f"Data cleaning completed. Shape after cleaning: {cleaned_data.shape}")
        return cleaned_data
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")

def clean_lat_long(data: pd.DataFrame, threshold=1):
    try:
        logger.info(f"Cleaning latitude and longitude values with threshold {threshold}")
        location_columns = ['restaurant_latitude',
                            'restaurant_longitude',
                            'delivery_latitude',
                            'delivery_longitude']

        result = (
            data.assign(**{
                col: (
                    np.where(data[col] < threshold, np.nan, data[col].values)
                )
                for col in location_columns
            })
        )

        nan_counts = {col: result[col].isna().sum() for col in location_columns}
        logger.info(f"NaN counts after lat/long cleaning: {nan_counts}")
        return result
    except Exception as e:
        logger.error(f"Error cleaning latitude and longitude values: {e}")

def calculate_haversine_distance(df):
    try:
        logger.info("Calculating haversine distance between restaurant and delivery locations")
        location_columns = ['restaurant_latitude',
                            'restaurant_longitude',
                            'delivery_latitude',
                            'delivery_longitude']

        # Check for NaN values in location columns
        nan_counts = {col: df[col].isna().sum() for col in location_columns}
        logger.debug(f"NaN counts in location columns: {nan_counts}")

        lat1 = df[location_columns[0]]
        lon1 = df[location_columns[1]]
        lat2 = df[location_columns[2]]
        lon2 = df[location_columns[3]]

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(
            dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

        c = 2 * np.arcsin(np.sqrt(a))
        distance = 6371 * c

        result = df.assign(distance=distance)
        logger.info(f"Distance calculation completed. Distance stats: min={result['distance'].min():.2f}, max={result['distance'].max():.2f}, mean={result['distance'].mean():.2f}")
        return result
    except Exception as e:
        logger.error(f"Error calculating haversine distance: {e}")

def create_distance_type(data: pd.DataFrame):
    try:
        logger.info("Creating distance type categories")
        result = (
            data
            .assign(
                distance_type=pd.cut(data["distance"], bins=[0, 5, 10, 15, 25],
                                     right=False, labels=["short", "medium", "long", "very_long"])
            )
        )

        # Log the distribution of distance types
        dist_counts = result['distance_type'].value_counts()
        logger.info(f"Distance type distribution: {dist_counts.to_dict()}")
        return result
    except Exception as e:
        logger.error(f"Error creating distance type categories: {e}")

def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    try:
        logger.info(f"Dropping {len(columns)} columns: {columns}")
        return df.drop(columns=columns)
    except Exception as e:
        logger.error(f"Error dropping columns: {e}")

def perform_data_cleaning(data: pd.DataFrame, saved_data_path: Path) -> pd.DataFrame:
    try:
        logger.info("Starting the data cleaning pipeline")
        
        cleaned_data = (
            data
            .pipe(change_column_names)
            .pipe(data_cleaning)
            .pipe(clean_lat_long)
            .pipe(calculate_haversine_distance)
            .pipe(create_distance_type)
            .pipe(drop_columns, columns=columns_to_drop)
        )
        
        # save the data
        logger.info(f"Saving cleaned data to {saved_data_path}")
        cleaned_data.to_csv(saved_data_path, index=False)
        logger.info(f"Data cleaning completed. Final shape: {cleaned_data.shape}")
        
        return cleaned_data
    except Exception as e:
        logger.error(f"Error during data cleaning pipeline: {e}")

if __name__ == "__main__":
    try:
        logger.info("Starting data cleaning script")
        
        # root path
        root_path = Path(__file__).parent.parent.parent
        logger.info(f"Root path: {root_path}")
        
        # data save directory
        cleaned_data_save_dir = root_path / "data" / "cleaned"
        # make directory if not exits
        cleaned_data_save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Cleaned data will be saved to: {cleaned_data_save_dir}")
        
        # cleaned data file name
        cleaned_data_filename = "swiggy_cleaned.csv"
        # data save path
        cleaned_data_save_path = cleaned_data_save_dir / cleaned_data_filename
        # data load path
        data_load_path = root_path / "data" / "raw" / "swiggy.csv"
        
        # load the data
        logger.info(f"Loading data from: {data_load_path}")
        df = load_data(data_load_path)

        # clean the data and save
        logger.info("Starting data cleaning process")
        perform_data_cleaning(data=df, saved_data_path=cleaned_data_save_path)
        logger.info("Data cleaning script completed successfully")
    except Exception as e:
        logger.error(f"An error occurred in the data cleaning script: {e}")
