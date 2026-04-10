# Predictive Maintenance — Vehicle Sensor Anomaly Detection

## Problem
Industrial equipment and vehicle fleets generate continuous sensor data. 
Detecting failures before they happen reduces downtime and maintenance costs. 
This project builds an end-to-end pipeline that ingests raw sensor data, 
engineers meaningful features, trains a machine learning model to predict 
failures, and visualizes patterns through an interactive dashboard.

## Dataset
AI4I 2020 Predictive Maintenance Dataset (UCI / Kaggle)

- 10,000+ sensor readings
- 3 machine types
- Multiple sensor signals (temperature, torque, speed, tool wear)
- Natural failure rate: ~3.39%

## What I Built

### 1. Data Pipeline
- Data loading and preprocessing
- Feature engineering (e.g., temperature difference, power load)
- Encoding categorical variables

### 2. Machine Learning Model
- Random Forest classifier
- SMOTE used to handle class imbalance
- Model evaluation using accuracy and ROC-AUC

### 3. Visualization Module
- Failure rate analysis by machine type
- Sensor behavior patterns during failures
- Feature importance visualization

### 4. Interactive Dashboard
- Built using Bokeh
- Visual exploration of failure trends
- Interactive charts saved as HTML dashboard

### 5. Deployment (MLOps)
- FastAPI-based REST API for real-time predictions
- JSON-based input for sensor values
- Returns failure probability and prediction label

### 6. Containerization
- Dockerized the entire ML service
- Portable deployment across environments
- Ready for cloud deployment (AWS/GCP/Render)
---  
## API Usage

### Endpoint
`POST /predict`

### Input Format
 ```json
{
  "Type_encoded": 1,
  "Air_temperature_K": 300,
  "Process_temperature_K": 310,
  "Rotational_speed_rpm": 1500,
  "Torque_Nm": 40,
  "Tool_wear_min": 120,
  "temp_diff": 10,
  "power": 5000
}
```

### Output Format
```json
{
  "failure_prediction": 0,
  "failure_probability": 0.49
}
```

## Key Results
- 98% classification accuracy on held-out test set
- ROC-AUC Score: 0.998
- Most predictive features: Tool wear, Torque, and Power load
- Class imbalance handled using SMOTE oversampling


## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Bokeh, FastAPI, Uvicorn, Docker, Matplotlib, Seaborn
