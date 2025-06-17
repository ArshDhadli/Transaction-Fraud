import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Create a simple fraud detection model
def create_model():
    # Generate synthetic training data
    np.random.seed(42)
    X_train = np.random.rand(100, 3)  # 3 features
    y_train = np.zeros(100)
    
    # Set some fraud patterns
    for i in range(20):  # 20% fraud rate
        idx = np.random.randint(0, 100)
        X_train[idx, 0] = np.random.uniform(0.8, 1.0)  # High amount
        X_train[idx, 1] = np.random.uniform(0.8, 1.0)  # High balance
        X_train[idx, 2] = 0  # Balance drops to 0
        y_train[idx] = 1
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model

# Main app
def main():
    st.title('Transaction Fraud Detection')
    
    # Create model
    model = create_model()
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input('Transaction Amount', min_value=0.0, value=1000.0)
        old_balance = st.number_input('Original Balance', min_value=0.0, value=5000.0)
    
    with col2:
        new_balance = st.number_input('New Balance', min_value=0.0, value=4000.0)
    
    # Normalize inputs
    amount_norm = amount / 10000  # Assuming max amount is 10000
    old_balance_norm = old_balance / 10000
    new_balance_norm = new_balance / 10000
    
    # Make prediction
    if st.button('Check Transaction'):
        features = np.array([[amount_norm, old_balance_norm, new_balance_norm]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        if prediction == 1:
            st.error(f"⚠️ This transaction is potentially fraudulent! (Probability: {probability:.2%})")
        else:
            st.success(f"✅ This transaction appears legitimate. (Probability: {probability:.2%})")

if __name__ == '__main__':
    main() 