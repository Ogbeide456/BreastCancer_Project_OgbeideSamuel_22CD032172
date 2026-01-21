
# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("Breast Cancer Prediction (Demo)")
st.write("This is an educational demo. Not a medical device.")

MODEL_PATH = "model/breast_cancer_model.pkl"

# Load model (show friendly error if missing)
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at '{MODEL_PATH}'. Run the training script or upload the file.")
    st.stop()

model = joblib.load(MODEL_PATH)

FEATURES = ['radius_mean','perimeter_mean','area_mean','smoothness_mean','concavity_mean']

st.sidebar.header("Input features")
inputs = {}
for f in FEATURES:
    # Use number_input with a broad range, adjust if you want min/max
    inputs[f] = st.sidebar.number_input(f, value=float( getattr(model, 'feature_default', 10.0) ), format="%.4f")

if st.sidebar.button("Predict"):
    X = np.array([inputs[f] for f in FEATURES]).reshape(1, -1)
    try:
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        label = "Malignant" if int(pred) == 1 else "Benign"
        st.success(f"Prediction: **{label}**")
        st.write(f"Probabilities — Benign: {proba[0]:.4f}, Malignant: {proba[1]:.4f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Make sure `model/breast_cancer_model.p

