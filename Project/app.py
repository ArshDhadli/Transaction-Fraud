import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Fraud Detection System", layout="wide", page_icon="🔍")

# Custom CSS
st.markdown("""
    <style>d
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

# Create a simple fraud detection model directly in the code
# This replaces loading the model from a file
feature_names = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                'newbalanceOrig', 'nameDest', 'oldbalanceDest', 
                'newbalanceDest', 'isFraud', 'isFlaggedFraud']

# Create a simple RandomForestClassifier model
fraud_model = RandomForestClassifier(n_estimators=10, random_state=42)

# Generate some synthetic training data
np.random.seed(42)
X_train = np.random.rand(100, 11)  # 100 samples, 11 features
y_train = np.zeros(100)

# Set some fraud patterns (high amount, balance drops to 0)
for i in range(20):  # 20% fraud rate
    idx = np.random.randint(0, 100)
    X_train[idx, 2] = np.random.uniform(5000, 10000)  # Higher amount
    X_train[idx, 4] = np.random.uniform(5000, 10000)  # High original balance
    X_train[idx, 5] = 0  # New balance drops to 0
    y_train[idx] = 1

# Train the model
fraud_model.fit(X_train, y_train)

# Store feature names in the model
fraud_model.feature_names_in_ = np.array(feature_names)

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

    # Data Visualizer Page
    elif selected == 'Data Visualizer':
        st.title('Fraud Detection Data Analysis')
        
        # Generate synthetic data for visualization instead of loading from file
        st.write("### Using Synthetic Data for Visualization")
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Create dataframe
        data = pd.DataFrame({
            'step': np.random.randint(1, 500, n_samples),
            'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'], n_samples),
            'amount': np.random.exponential(1000, n_samples),
            'oldbalanceOrg': np.random.exponential(5000, n_samples),
            'newbalanceOrig': np.zeros(n_samples),
            'oldbalanceDest': np.random.exponential(2000, n_samples),
            'newbalanceDest': np.zeros(n_samples),
            'isFraud': np.zeros(n_samples, dtype=int),
            'isFlaggedFraud': np.zeros(n_samples, dtype=int)
        })
        
        # Set newbalanceOrig based on transaction
        for i in range(n_samples):
            # Normal transactions: balance decreases by amount but stays >= 0
            data.loc[i, 'newbalanceOrig'] = max(0, data.loc[i, 'oldbalanceOrg'] - data.loc[i, 'amount'])
            # Destination balance increases by amount
            data.loc[i, 'newbalanceDest'] = data.loc[i, 'oldbalanceDest'] + data.loc[i, 'amount']
        
        # Create fraud patterns (~10% of transactions)
        fraud_indices = np.random.choice(n_samples, size=int(n_samples*0.1), replace=False)
        for idx in fraud_indices:
            data.loc[idx, 'amount'] = np.random.uniform(5000, 10000)  # Higher amount
            data.loc[idx, 'oldbalanceOrg'] = np.random.uniform(5000, 10000)  # High original balance
            data.loc[idx, 'newbalanceOrig'] = 0  # New balance drops to 0
            data.loc[idx, 'isFraud'] = 1
            
            # Flag some of the fraudulent transactions (about 20%)
            if np.random.random() < 0.2:
                data.loc[idx, 'isFlaggedFraud'] = 1
        
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

if __name__ == '__main__':
    main() 