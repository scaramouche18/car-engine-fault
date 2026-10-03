import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/engine_fault_model.pkl')

st.title("Car Engine Fault Detection")
st.write("Enter sensor readings below to predict engine condition.")

# Input fields for each sensor
vibration_amplitude = st.slider("Vibration Amplitude", 0.0, 10.0, 5.0)
rms_vibration = st.slider("RMS Vibration", 0.0, 5.0, 2.5)
vibration_frequency = st.slider("Vibration Frequency", 0.0, 2000.0, 1000.0)
surface_temp = st.slider("Surface Temperature", 30.0, 150.0, 90.0)
exhaust_temp = st.slider("Exhaust Temperature", 200.0, 600.0, 400.0)
acoustic_db = st.slider("Acoustic dB", 60.0, 120.0, 90.0)
acoustic_freq = st.slider("Acoustic Frequency", 100.0, 5000.0, 2500.0)
intake_pressure = st.slider("Intake Pressure", 90.0, 120.0, 105.0)
exhaust_pressure = st.slider("Exhaust Pressure", 80.0, 110.0, 95.0)
freq_band_energy = st.slider("Frequency Band Energy", 0.1, 1.0, 0.5)
amplitude_mean = st.slider("Amplitude Mean", 0.01, 0.5, 0.25)

# Predict button
if st.button("Predict Engine Condition"):
    input_data = pd.DataFrame([[
        vibration_amplitude, rms_vibration, vibration_frequency,
        surface_temp, exhaust_temp, acoustic_db, acoustic_freq,
        intake_pressure, exhaust_pressure, freq_band_energy, amplitude_mean
    ]], columns=[
        'Vibration_Amplitude', 'RMS_Vibration', 'Vibration_Frequency',
        'Surface_Temperature', 'Exhaust_Temperature', 'Acoustic_dB',
        'Acoustic_Frequency', 'Intake_Pressure', 'Exhaust_Pressure',
        'Frequency_Band_Energy', 'Amplitude_Mean'
    ])

    prediction = model.predict(input_data)[0]

    labels = {0: "Normal ✅", 1: "Minor Fault ⚠️", 2: "Critical Fault 🚨"}
    st.subheader(f"Prediction: {labels[prediction]}")
