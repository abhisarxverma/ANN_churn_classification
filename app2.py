import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os

# ========== 🚀 Load Model ========== #
if os.path.exists("model.h5"):
    model = load_model("model.h5", compile=False)  # Load without compiling to avoid optimizer issues
    model.build(input_shape=(None, 11))  # Ensure correct input shape (adjust as per your model's features)
else:
    st.error("Model file 'model.h5' not found!")
    st.stop()

# Recompile the model with proper optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# ========== 🔄 Load Preprocessing Objects ========== #
try:
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading encoders or scaler: {e}")
    st.stop()

# ========== 🎨 Streamlit UI ========== #
st.title("🔮 Customer Churn Prediction")

# User Inputs
geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox('🧑‍🤝‍🧑 Gender', label_encoder_gender.classes_)
age = st.slider('📅 Age', 18, 92)
balance = st.number_input('💰 Balance')
credit_score = st.number_input('📊 Credit Score')
estimated_salary = st.number_input("💵 Estimated Salary")
tenure = st.slider('⏳ Tenure', 0, 10)
num_of_products = st.slider('📦 Number of Products', 1, 4)
has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

# ========== 🔄 Data Preprocessing ========== #
try:
    # Convert gender using LabelEncoder
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Prepare the DataFrame
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform(np.array([[geography]])).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Merge with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Ensure feature order matches training data
    expected_features = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else None
    if expected_features is not None:
        input_data = input_data[expected_features]

    # Apply Scaling
    input_data_scaled = scaler.transform(input_data)

except Exception as e:
    st.error(f"Error in data preprocessing: {e}")
    st.stop()

# ========== 🔮 Prediction ========== #
try:
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.write(f"### 🤖 Churn Probability: **{prediction_proba:.4f}**")

    if prediction_proba > 0.5:
        st.error("🚨 The customer is **likely to churn**!")
    else:
        st.success("✅ The customer is **not likely to churn**.")

except Exception as e:
    st.error(f"Prediction error: {e}")
