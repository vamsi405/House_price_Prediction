import streamlit as st
import pickle
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# -------------------------
# Train model if not exists
# -------------------------
if not os.path.exists("model.pkl"):
    data = fetch_california_housing()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

# -------------------------
# Load model
# -------------------------
model = pickle.load(open("model.pkl", "rb"))

st.title("🏠 House Price Prediction App")

st.write("Enter house details below:")

MedInc = st.number_input("Median Income", min_value=0.0)
HouseAge = st.number_input("House Age", min_value=0.0)
AveRooms = st.number_input("Average Rooms", min_value=0.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0)
Population = st.number_input("Population", min_value=0.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.0)
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

if st.button("Predict Price"):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])
    
    prediction = model.predict(features)
    st.success(f"Predicted House Price: ${prediction[0]*100000:.2f}")

