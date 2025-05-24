# Fraud Detection System

A web application for detecting fraudulent transactions using machine learning.

## Features

- Predict whether a transaction is fraudulent based on transaction details
- Visualize transaction data and patterns
- User-friendly interface built with Streamlit

## Setup Instructions

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open the app in your browser:
   The app will typically be available at http://localhost:8501

## Project Structure

- `app.py`: Main Streamlit application
- `fraud_detection_model.pkl`: Trained machine learning model
- `synthetic_data.csv`: Dataset containing transaction information
- `requirements.txt`: List of Python package dependencies

## How to Use

1. Navigate to the "Fraud Detection" tab in the sidebar
2. Enter transaction details in the form
3. Click "Check Transaction" to get a prediction
4. Use the "Data Visualizer" tab to explore patterns in the data

## Model Information

This application uses a RandomForest classifier trained on synthetic financial transaction data to identify potentially fraudulent transactions. 