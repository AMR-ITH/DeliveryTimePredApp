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

swiggy_order_time_prediction
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
