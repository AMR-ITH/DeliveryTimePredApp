## 🍕 Delivery Time Prediction for Food Chain App

### 📋 Project Overview
This project develops a machine learning pipeline to predict delivery times for a food chain app. The workflow includes data cleaning, exploratory data analysis (EDA), preprocessing, base model building, hyperparameter tuning with Optuna, stacking, model tracking with MLflow, data versioning with DVC, and deployment via a CI/CD pipeline.

- **Data Pipeline**: Cleaning, EDA, preprocessing, training, evaluation, and model registration.
- **Models**: Base models (KNN, LR) optimized with Optuna, stacked with LR as the final estimator.
- **Tracking**: Experiments logged in MLflow, with the final model saved in staging.
- **Data Versioning**: DVC pipeline with data stored in an S3-backed DVC repository.
- **Deployment**: CI/CD pipeline using AWS CodeDeploy and ECR, with FastAPI for predictions.

### 🛠️ Tech Stack
#### Data & Pipeline
- **DVC (Data Version Control)**: Versioning datasets and managing the data pipeline.
- **AWS S3**: Storage for datasets and pipeline artifacts.

#### CI/CD
- **GitHub Actions**: Automates CI/CD pipeline.
- **Docker**: Containerizes the application.
- **AWS ECR (Elastic Container Registry)**: Stores Docker images.
- **AWS CodeDeploy**: Deploys the application.

#### Deployment
- **AWS EC2/ECS**: Hosts the deployed application.
- **Auto Scaling Groups (Rolling Update)**: Ensures scalability and zero-downtime updates.
- **FastAPI**: Serves model predictions via a REST API.

## 📊 System Overview
### Image 1: Machine Learning Workflow with Version Control
![image](https://github.com/user-attachments/assets/e3981564-03e9-4e5b-be69-6ae9dc838b1e)

### Image 2: CI/CD and Deployment Process
![image](https://github.com/user-attachments/assets/e3e3377e-6948-473a-8b0a-0975c2ff679b)


## 🌐 Try It Out
You can interact with the recommendation system at the following link:  
(http:[http://13.201.71.93/docs])

## 📋 Example Input Data
You can download an example input dataset in CSV format from the following link:  

    {
      "ID": "0x4607",
      "Delivery_person_ID": "INDORES13DEL02",
      "Delivery_person_Age": 37.0,
      "Delivery_person_Ratings": 4.9,
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
      "multiple_deliveries": 0,
      "Festival": "no",
      "City": "Urban"
    }


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
