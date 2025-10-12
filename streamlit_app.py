import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
st.title("🎈  dl prediction model")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
df = pd.read_csv("https://raw.githubusercontent.com/subashjeerla/dl_model_iiit/refs/heads/main/real_2015")
df


# Load trained model and scalers
# model = load_model("pm25_model.h5")
# scaler_X = joblib.load("scaler_X.pkl")
# scaler_y = joblib.load("scaler_y.pkl")

# App title and description
st.title("🌫️ PM2.5 Air Pollution Prediction")
st.write("Enter the weather details below to predict the PM2.5 concentration.")

# User inputs (same as your dataset columns except target)
T = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0)
TM = st.number_input("Max Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
Tm = st.number_input("Min Temperature (°C)", min_value=-20.0, max_value=40.0, value=10.0)
SLP = st.number_input("Sea Level Pressure (hPa)", min_value=950.0, max_value=1050.0, value=1015.0)
H = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
VV = st.number_input("Visibility (km)", min_value=0.0, max_value=10.0, value=1.0)
V = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=2.0)
VM = st.number_input("Max Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=5.0)

# Predict button
# if st.button("🔮 Predict PM2.5"):
#     # Prepare input
#     user_input = np.array([[T, TM, Tm, SLP, H, VV, V, VM]])
#     user_input_scaled = scaler_X.transform(user_input)
    
#     # Predict
#     pred_scaled = model.predict(user_input_scaled)
#     pred_original = scaler_y.inverse_transform(pred_scaled)
#     pm_value = float(pred_original[0][0])

#     st.success(f"### Predicted PM2.5 Value: {pm_value:.2f} µg/m³")

#     # Air Quality Category
#     if pm_value <= 50:
#         category = "Good 😊"
#         color = "green"
#     elif pm_value <= 100:
#         category = "Moderate 😐"
#         color = "yellow"
#     elif pm_value <= 150:
#         category = "Unhealthy for Sensitive Groups 😷"
#         color = "orange"
#     elif pm_value <= 200:
#         category = "Unhealthy 😣"
#         color = "red"
#     elif pm_value <= 300:
#         category = "Very Unhealthy 😫"
#         color = "purple"
#     else:
#         category = "Hazardous ☠️"
#         color = "brown"

#     st.markdown(f"<h3 style='color:{color};'>Air Quality: {category}</h3>", unsafe_allow_html=True)
