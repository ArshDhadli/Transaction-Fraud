import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import re

# Set page configuration
st.set_page_config(page_title="Fraud Detection System", layout="wide", page_icon="🔍")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stPlotlyChart {
        width: 100%;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained fraud detection model
try:
    fraud_model = joblib.load('fraud_detection_model.pkl')
    # Get expected feature names from the model if available
    feature_names = fraud_model.feature_names_in_ if hasattr(fraud_model, 'feature_names_in_') else None
    if feature_names is not None:
        st.write(f"Model expects these features: {', '.join(feature_names)}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    fraud_model = None
    feature_names = None

# Function for the main logic of the app
def main():
    # Sidebar for navigation
    with st.sidebar:
        selected = option_menu(
            'Fraud Detection System',
            ['Data Visualizer', 'Fraud Detection'],
            icons=['bar-chart', 'shield-exclamation'],
            default_index=0
        )

    # Fraud Detection Page
    if selected == 'Fraud Detection':
        st.title('Transaction Fraud Detection using ML')
        st.write("Enter transaction details to check for potential fraud")

        # Columns for input fields
        col1, col2, col3 = st.columns(3)

        with col1:
            step = st.number_input('Step (time step)', min_value=1, value=1)
            transaction_type = st.selectbox('Transaction Type', 
                                           ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
            amount = st.number_input('Amount', min_value=0.0, value=1000.0)
            
        with col2:
            old_balance_orig = st.number_input('Origin Account Original Balance', min_value=0.0, value=5000.0)
            new_balance_orig = st.number_input('Origin Account New Balance', min_value=0.0, value=4000.0)
            
        with col3:
            old_balance_dest = st.number_input('Destination Account Original Balance', min_value=0.0, value=1000.0)
            new_balance_dest = st.number_input('Destination Account New Balance', min_value=0.0, value=2000.0)
            is_flagged_fraud = st.selectbox('Is Flagged as Fraud by System', [0, 1], index=0)

        # Code for prediction
        fraud_diagnosis = ''
        
        # Creating a button for prediction
        if st.button('Check Transaction'):
            if fraud_model is not None:
                # Convert transaction type to one-hot encoding if needed
                type_mapping = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
                type_encoded = type_mapping.get(transaction_type, 0)
                
                # Create feature array with the correct dimensions (11 features expected)
                # Note: We're handling only the numerical features
                # Adding default values for any other expected features
                features = np.zeros(11)  # Initialize with zeros for all 11 features
                
                # Fill in the features we have
                # Adjust these indices based on your model's expected feature order
                features[0] = step  # step
                features[1] = type_encoded  # type
                features[2] = amount  # amount
                # Skip nameOrig (index 3)
                features[4] = old_balance_orig  # oldbalanceOrg
                features[5] = new_balance_orig  # newbalanceOrig
                # Skip nameDest (index 6)
                features[7] = old_balance_dest  # oldbalanceDest
                features[8] = new_balance_dest  # newbalanceDest
                # Index 9 is probably isFraud (target variable) - we don't need to include it
                features[10] = is_flagged_fraud  # isFlaggedFraud
                
                # Reshape to 2D array for sklearn
                features = features.reshape(1, -1)
                
                # Make prediction
                try:
                    # Check if we're providing the right number of features
                    st.write(f"Providing {features.shape[1]} features to the model")
                    
                    fraud_prediction = fraud_model.predict(features)
                    
                    if fraud_prediction[0] == 1:
                        fraud_diagnosis = "⚠️ This transaction is potentially fraudulent!"
                        st.error(fraud_diagnosis)
                    else:
                        fraud_diagnosis = "✅ This transaction appears legitimate."
                        st.success(fraud_diagnosis)
                    
                    # Display confidence if model has predict_proba method
                    if hasattr(fraud_model, 'predict_proba'):
                        fraud_prob = fraud_model.predict_proba(features)[0][1]
                        st.write(f"Fraud probability: {fraud_prob:.2%}")
                        
                        # Create a gauge chart for fraud probability
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = fraud_prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Fraud Probability"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkred"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgreen"},
                                    {'range': [25, 50], 'color': "yellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "red"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig)
                        
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.write("Debugging info:")
                    st.write(f"Features shape: {features.shape}")
                    st.write("Features:")
                    st.write(features)
            else:
                st.error("Model could not be loaded. Please check if the model file exists.")

    # Data Visualizer Page
    elif selected == 'Data Visualizer':
        st.title('Fraud Detection Data Analysis')
        
        # Load sample data for visualization
        try:
            # Attempt to load the first 10,000 rows for visualization
            # Adjust based on your actual data size
            data = pd.read_csv('synthetic_data.csv', nrows=10000)
            
            st.write("### Data Overview")
            st.dataframe(data.head())
            
            st.write("### Transaction Type Distribution")
            fig1 = px.histogram(data, x='type', color='isFraud', 
                              barmode='group', 
                              color_discrete_map={0: 'green', 1: 'red'},
                              labels={'type': 'Transaction Type', 'isFraud': 'Is Fraud'})
            st.plotly_chart(fig1, use_container_width=True)
            
            st.write("### Transaction Amount Distribution by Fraud Status")
            fig2 = px.box(data, x='isFraud', y='amount', 
                        color='isFraud',
                        color_discrete_map={0: 'green', 1: 'red'},
                        labels={'isFraud': 'Is Fraud (0=No, 1=Yes)', 'amount': 'Transaction Amount'})
            st.plotly_chart(fig2, use_container_width=True)
            
            st.write("### Balance Changes in Fraudulent vs. Legitimate Transactions")
            # Calculate balance delta
            data['orig_balance_delta'] = data['newbalanceOrig'] - data['oldbalanceOrg']
            data['dest_balance_delta'] = data['newbalanceDest'] - data['oldbalanceDest']
            
            # Create subplot with two histograms
            fig3 = make_subplots(rows=1, cols=2, 
                                subplot_titles=("Origin Account Balance Change", 
                                                "Destination Account Balance Change"))
            
            # Add traces for origin balance delta
            fig3.add_trace(
                go.Histogram(x=data[data['isFraud']==0]['orig_balance_delta'], 
                           name="Legitimate", marker_color='green', opacity=0.7),
                row=1, col=1
            )
            fig3.add_trace(
                go.Histogram(x=data[data['isFraud']==1]['orig_balance_delta'], 
                           name="Fraudulent", marker_color='red', opacity=0.7),
                row=1, col=1
            )
            
            # Add traces for destination balance delta
            fig3.add_trace(
                go.Histogram(x=data[data['isFraud']==0]['dest_balance_delta'], 
                           name="Legitimate", marker_color='green', showlegend=False, opacity=0.7),
                row=1, col=2
            )
            fig3.add_trace(
                go.Histogram(x=data[data['isFraud']==1]['dest_balance_delta'], 
                           name="Fraudulent", marker_color='red', showlegend=False, opacity=0.7),
                row=1, col=2
            )
            
            fig3.update_layout(barmode='overlay')
            st.plotly_chart(fig3, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading or visualizing data: {e}")
            st.info("Upload a CSV file with fraud detection data to visualize")
            
            # Allow user to upload their own data
            uploaded_file = st.file_uploader("Upload your fraud detection dataset (CSV)", type="csv")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.success("Data loaded successfully!")
                st.dataframe(data.head())

if __name__ == '__main__':
    main() 