## 🍕 Delivery Time Prediction for Food Chain App

# ML Delivery Time Prediction System

## 📋 Project Overview

This project develops an end-to-end machine learning pipeline to predict delivery times for a food delivery app. 
The system handles everything from data processing to model deployment with comprehensive experiment tracking and versioning.

### Key Components
- **Data Pipeline**: Data cleaning, feature engineering, preprocessing and training pipeline
- **Model Development**: Systematic experimentation and model selection
- **MLflow Integration**: Experiment tracking and model registry
- **DVC Pipeline**: Data versioning and pipeline automation
- **CI/CD Deployment**: Automated testing and deployment

## 🔄 Workflow Architecture
![image](https://github.com/user-attachments/assets/5d625152-9924-4f22-851f-5cda4c4f88ef)

The diagram above illustrates our end-to-end workflow:
1. Data and code are version-controlled in repositories (Notebook, SRC, Data)
2. Experiments are tracked via MLflow (Experiment Tracking)
3. Model registry maintains production-ready models
4. DVC pipelines automate the workflow
5. Best models are deployed via FastAPI for predictions
6. All components are integrated with version control (Git, GitHub)
7. Data is stored and versioned in AWS S3
8. DagShub serves as the central hub for project management

## 🧪 ML Experiment Tracking

All experiments are tracked in MLflow with the following sequence:

1. **Exp 1 - Keep Vs Drop Missing Values**: Compared model performance with and without missing value imputation to establish baseline data preprocessing strategy.

2. **Exp 2 - Model Selection**: Evaluated multiple algorithms (KNN, LR, etc.) with initial hyperparameter tuning via Optuna to identify the most promising models.

3. **Exp 3 - RF HP Tuning**: Optimized Random Forest hyperparameters using Optuna for improved accuracy.

4. **Exp 4 - LGBM HP Tuning**: Fine-tuned LightGBM hyperparameters with Optuna for better performance.

5. **Exp 5 - Final Estimator**: Built and saved the stacked ensemble model using the best-performing models with LR as the final estimator.

## 📊 Model Performance

The final stacked model achieved the following performance metrics:

**Training Metrics:**
- MAE: 2.55
- R²: 0.88

**Testing Metrics:**
- MAE: 3.06
- R²: 0.84

**Cross-Validation:**
- CV Score (MAE): 3.08

These metrics indicate strong predictive performance with good generalization between training and testing datasets.

## 🛠️ Tech Stack

### Data & Pipeline
- **DVC (Data Version Control)**: Versioning datasets and managing the data pipeline
- **MLflow**: Experiment tracking and model registry
- **AWS S3**: Storage for datasets and pipeline artifacts
- **Optuna**: Hyperparameter optimization

### CI/CD
- **GitHub Actions**: Automating testing and deployment
- **Docker**: Containerizing the application
- **AWS ECR**: Storing Docker images
- **AWS CodeDeploy**: Managing deployment processes

### Deployment
- **AWS EC2/ECS**: Hosting the production application
- **Auto Scaling Groups**: Ensuring scalability with rolling updates for zero downtime
- **FastAPI**: Serving model predictions via REST API
- **Dagshub**: Central hub for ML project management



## 🔍 API Usage

The deployed model is accessible via a REST API. You can interact with it at:
http://13.201.71.93/docs

### Example API Request

```python
import requests
import json

url = "http://13.201.71.93/predict"
payload = {
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

response = requests.post(url, json=payload)
print(response.json())
```

### Example Input Format

```json
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
```

## 🔄 CI/CD Pipeline

![image](https://github.com/user-attachments/assets/5fe21c6d-9f4c-45c5-9b71-18775c96d9dc)


CI/CD pipeline automates the entire deployment process:

1. **GitHub Actions**: Automates CI/CD pipeline triggering tests and builds upon code changes
2. **Docker**: Containerizes the application for consistent deployment
3. **AWS ECR**: Stores Docker images securely in the cloud
4. **AWS EC2/ECS**: Hosts the deployed application in a scalable environment
5. **AWS CodeDeploy**: Manages the application deployment process
6. **FastAPI**: Serves model predictions via a RESTful API
7. **Auto Scaling Groups**: Ensures scalability and zero-downtime updates as traffic demands

This architecture allows for seamless deployment from development to production with minimal manual intervention.



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
