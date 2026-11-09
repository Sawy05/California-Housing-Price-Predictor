# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")
train_encoded_cols = joblib.load("train_encoded_cols.pkl")  # Load the columns used during training

st.title("California Housing Price Predictor")
st.write("Enter property details to predict median house value:")

# Input fields
input_data = {
    "longitude": st.number_input("Longitude", value=-118.0),
    "latitude": st.number_input("Latitude", value=34.0),
    "housing_median_age": st.number_input("Housing Median Age", min_value=1, value=30),
    "total_rooms": st.number_input("Total Rooms", min_value=1, value=2000),
    "total_bedrooms": st.number_input("Total Bedrooms", min_value=1, value=400),
    "population": st.number_input("Population", min_value=1, value=1000),
    "households": st.number_input("Households", min_value=1, value=400),
    "median_income": st.number_input("Median Income", min_value=0.0, value=4.0),
    "ocean_proximity_<1H OCEAN": 0,
    "ocean_proximity_INLAND": 0,
    "ocean_proximity_ISLAND": 0,
    "ocean_proximity_NEAR BAY": 0,
    "ocean_proximity_NEAR OCEAN": 0,
}

# Ocean proximity dropdown
proximity = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])
input_data[f"ocean_proximity_{proximity}"] = 1

# Convert to DataFrame
df_input = pd.DataFrame([input_data])

# Align columns with training columns
for col in train_encoded_cols:
    if col not in df_input.columns:
        df_input[col] = 0

df_input = df_input[train_encoded_cols]

# Predict
if st.button("Predict"):
    scaled = scaler.transform(df_input)
    prediction = model.predict(scaled)
    st.success(f"Estimated Median House Value: ${prediction[0]:,.2f}")
