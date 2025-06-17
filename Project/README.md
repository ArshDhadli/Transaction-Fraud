# Transaction Fraud Detection App

A Streamlit application that helps detect potentially fraudulent transactions based on transaction amount and account balance patterns.

## Features
- Real-time fraud detection
- User-friendly interface
- Risk score calculation
- Detailed transaction analysis

## Deployment
This app is deployed on Streamlit Cloud. You can access it at: [Your Streamlit URL will appear here after deployment]

## Local Development
To run this app locally:

1. Clone the repository
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run app.py
```

## Requirements
- Python 3.7+
- Streamlit
- NumPy
- scikit-learn

## Project Structure
```
Project/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## How to Use

1. Navigate to the "Fraud Detection" tab in the sidebar
2. Enter transaction details in the form
3. Click "Check Transaction" to get a prediction
4. Use the "Data Visualizer" tab to explore patterns in the data

## Model Information

This application uses a RandomForest classifier trained on synthetic financial transaction data to identify potentially fraudulent transactions. 