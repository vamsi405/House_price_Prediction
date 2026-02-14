import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model.pkl"
DATA_PATH = "data/housing.csv"

# -------------------------
# Train model if not exists
# -------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline


if not os.path.exists(MODEL_PATH):
    model = train_model()
else:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


# -------------------------
# STREAMLIT UI
# -------------------------

st.title("🏠 House Price Prediction")

st.write("Enter house details below:")

df = pd.read_csv(DATA_PATH)
feature_columns = df.drop("price", axis=1).columns

input_data = []

for col in feature_columns:
    value = st.number_input(f"{col}")
    input_data.append(value)

if st.button("Predict Price"):
    features = np.array([input_data])
    prediction = model.predict(features)
    st.success(f"Predicted Price: {prediction[0]:,.2f}")
