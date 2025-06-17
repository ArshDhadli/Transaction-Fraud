import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="centered"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

def create_model():
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    X = np.random.rand(n_samples, 2)  # 2 features: amount and balance
    y = np.zeros(n_samples)
    
    # Create fraud patterns
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    X[fraud_indices, 0] = np.random.uniform(0.8, 1.0, size=len(fraud_indices))  # High amounts
    X[fraud_indices, 1] = np.random.uniform(0, 0.2, size=len(fraud_indices))    # Low balances
    y[fraud_indices] = 1
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def main():
    st.title('🔍 Transaction Fraud Detection')
    st.markdown("""
    This application helps detect potentially fraudulent transactions based on 
    transaction amount and account balance patterns.
    """)
    
    # Create model
    model = create_model()
    
    # Input fields with better formatting
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input(
            'Transaction Amount ($)',
            min_value=0.0,
            value=1000.0,
            step=100.0,
            format="%.2f"
        )
    
    with col2:
        balance = st.number_input(
            'Account Balance ($)',
            min_value=0.0,
            value=5000.0,
            step=100.0,
            format="%.2f"
        )
    
    # Normalize inputs
    amount_norm = amount / 10000  # Assuming max amount is 10000
    balance_norm = balance / 10000
    
    # Make prediction
    if st.button('Check Transaction'):
        features = np.array([[amount_norm, balance_norm]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Display results with better formatting
        st.markdown("---")
        if prediction == 1:
            st.error(f"""
            ⚠️ **Potentially Fraudulent Transaction**
            
            Risk Score: {probability:.1%}
            
            This transaction shows patterns commonly associated with fraudulent activity.
            """)
        else:
            st.success(f"""
            ✅ **Legitimate Transaction**
            
            Risk Score: {probability:.1%}
            
            This transaction appears to be legitimate based on the patterns analyzed.
            """)
        
        # Add explanation
        st.markdown("""
        ### How it works
        The system analyzes:
        - Transaction amount relative to typical amounts
        - Account balance patterns
        - Historical fraud patterns
        
        Higher risk scores indicate a higher probability of fraudulent activity.
        """)

if __name__ == '__main__':
    main() 