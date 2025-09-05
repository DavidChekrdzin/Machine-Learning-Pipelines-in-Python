# Vehicle CO2 Emission Prediction (Random Forest + Pipelines)

This project builds a machine learning model to predict **CO2 emissions** of vehicles based on their specifications.  
It demonstrates the use of **data preprocessing pipelines**, **feature encoding**, and **Random Forest regression** for building an end-to-end machine learning workflow.

---

## 🚀 Features
- Preprocesses data using **pipelines**:
  - Handles missing values
  - Standardizes numerical features
  - Encodes categorical features with One-Hot Encoding
- Splits dataset into training and testing sets
- Trains a **Random Forest Regressor**
- Evaluates the model with:
  - **R² score**
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Error (MAE)**
- Saves the trained model as a `.joblib` file for reuse

---

## 📂 Project Structure
├── vehicle_emissions.csv # Dataset file (add it yourself)

├── co2_emission_prediction.py # Main Python script

├── vehicle_emission_pipeline.joblib # Saved trained pipeline

├── README.md # Project documentation

## ⚙️ Installation
1. Clone this repository
2. Install dependencies:
   
      Python 3.8+
     
      pandas
     
      numpy
     
      scikit-learn
     
      joblib
