import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model_xgb_top10_adm_sl.pkl')

# Define input features
features = [
    'adm_value ALL_AGRATIO', 'Slope SGPT', 'Slope CA', 'Slope ALP',
    'adm_value SGPT', 'adm_value MCHC', 'Slope WBC', 'Slope BUN',
    'adm_value LAC', 'Slope RDWCV'
]

# Streamlit UI
st.title("AKI Prediction Model")
st.write("Enter the feature values to predict the outcome.")

# Create input fields for 10 features
inputs = []
for feature in features:
    value = st.number_input(f"{feature}", step=0.01, format="%.2f")
    inputs.append(value)

# Predict button
if st.button("Predict"):
    try:
        # Convert inputs to numpy array
        input_array = np.array(inputs).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)

        # Display prediction result
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
