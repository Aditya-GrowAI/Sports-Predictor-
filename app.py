import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the scaler
model, scaler = None, None
try:
    model = joblib.load('sport_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'sport_model.pkl' and 'scaler.pkl' are in the same directory.")

# Set up the Streamlit app layout
st.title('Which Sport Are You Good At? 🏃‍♀️')
st.write('Enter your physical attributes below to predict which sport you might excel in.')

# Create input fields for user
st.sidebar.header('Enter Your Attributes')

stride_length = st.sidebar.slider('Stride Length (cm)', 50, 200, 100)
height = st.sidebar.slider('Height (ft)', 4.0, 7.0, 5.5, step=0.1)
speed = st.sidebar.slider('Speed (kmph)', 5, 40, 15)
jump_height = st.sidebar.slider('Jump Height (cm)', 10, 150, 40)
endurance = st.sidebar.slider('Endurance (min)', 10, 180, 60)

# Create a DataFrame from the user inputs
user_data = pd.DataFrame({
    'Stride_length_cm': [stride_length],
    'Height_ft': [height],
    'Speed_kmph': [speed],
    'Jump_height_cm': [jump_height],
    'Endurance_min': [endurance]
})

# Display the user input
st.subheader('Your Entered Attributes')
st.write(user_data)

# Only proceed if model and scaler are loaded
if model is not None and scaler is not None:
    user_data_scaled = scaler.transform(user_data)
    if st.button('Predict Sport'):
        try:
            prediction = model.predict(user_data_scaled)
            st.success(f'Based on your attributes, you might be good at: **{prediction[0]}** 🎉')
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Prediction is unavailable until model and scaler files are present.")