import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("model.joblib")
columns = joblib.load("columns.joblib")

st.set_page_config(page_title="Fraud Detection System", layout="centered")

st.title("Fraud Detection System")
st.write("""
This app predicts whether a transaction is **Fraudulent** or **Legitimate** 
based on the details you provide below.
""")

st.markdown("---")
st.header("Transaction Details")

# Country mapping
country_map = {
    "Germany": "DE",
    "Spain": "ES",
    "France": "FR",
    "United Kingdom": "GB",
    "Italy": "IT",
    "Netherlands": "NL",
    "Poland": "PL",
    "Romania": "RO",
    "Turkey": "TR",
    "United States": "US"
}

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    account_age_days = st.number_input("Account Age (Days)", min_value=0, max_value=5000, value=365)
    total_transactions_user = st.number_input("Total Transactions (User)", min_value=0, value=50)
    avg_amount_user = st.number_input("Average Transaction Amount", min_value=0.0, value=200.0)
    amount = st.number_input("Current Transaction Amount", min_value=0.0, value=100.0)
    shipping_distance_km = st.number_input("Shipping Distance (km)", min_value=0.0, value=10.0)

with col2:
    country_full = st.selectbox("Country", list(country_map.keys()))
    bin_country_full = st.selectbox("BIN Country", list(country_map.keys()))
    channel = st.selectbox("Channel", ['App', 'Web'])
    merchant_category = st.selectbox("Merchant Category", ['Electronics', 'Fashion', 'Gaming', 'Grocery', 'Travel'])
    promo_used = st.selectbox("Promo Used", ['No', 'Yes'])

st.markdown("---")
st.header("Security Checks")

col3, col4, col5 = st.columns(3)
with col3:
    avs_match = st.selectbox("AVS Match", ['No', 'Yes'])
with col4:
    cvv_result = st.selectbox("CVV Result", ['No', 'Yes'])
with col5:
    three_ds_flag = st.selectbox("3D Secure Flag", ['No', 'Yes'])

st.markdown("---")

if st.button("Predict Fraud"):
    # Convert selections back to numeric or coded values
    country = country_map[country_full]
    bin_country = country_map[bin_country_full]
    channel = channel.lower()
    merchant_category = merchant_category.lower()
    promo_used = 1 if promo_used == 'Yes' else 0
    avs_match = 1 if avs_match == 'Yes' else 0
    cvv_result = 1 if cvv_result == 'Yes' else 0
    three_ds_flag = 1 if three_ds_flag == 'Yes' else 0

    # Prepare input as DataFrame
    input_data = {
        'account_age_days': [account_age_days],
        'total_transactions_user': [total_transactions_user],
        'avg_amount_user': [avg_amount_user],
        'amount': [amount],
        'promo_used': [promo_used],
        'avs_match': [avs_match],
        'cvv_result': [cvv_result],
        'three_ds_flag': [three_ds_flag],
        'shipping_distance_km': [shipping_distance_km],
        'country': [country],
        'bin_country': [bin_country],
        'channel': [channel],
        'merchant_category': [merchant_category]
    }

    df = pd.DataFrame(input_data)

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=['country', 'bin_country', 'channel', 'merchant_category'], drop_first=False)

    # Add missing columns and align order
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[columns]

    # Make prediction
    prediction = model.predict(df_encoded)[0]
    probability = model.predict_proba(df_encoded)[0][1]  # Probability of fraud

    # Display styled result
    st.markdown("### Prediction Result:")

    if prediction == 1:
        st.markdown(
            f"""
            <div style='background-color:#4a0e0e; padding:15px; border-radius:10px;'>
                <p style='color:#ff4b4b; font-size:18px;'>
                The transaction is <b>likely FRAUDULENT</b> (Probability: {probability:.2f})
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='background-color:#0e3b27; padding:15px; border-radius:10px;'>
                <p style='color:#21c46d; font-size:18px;'>
                The transaction is <b>likely LEGITIMATE</b> (Probability: {1 - probability:.2f})
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- Model credit line ---
st.markdown(
    """
    <hr style='margin-top:40px;'>
    <p style='text-align:center; color:gray; font-size:14px;'>
        Model used: <b>XGBoost Classifier</b> | Balanced using <b>SMOTE</b> & <b>Class Weights</b>
    </p>
    """,
    unsafe_allow_html=True
)

# --- Footer credit line ---
st.markdown(
    """
    <p style='text-align:center; color:gray; font-size:14px; margin-top:10px;'>
        © 2025 Fraud Detection System | Built by <b>Tamin</b> with Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
