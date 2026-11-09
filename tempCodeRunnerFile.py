streamlit_code = """
import streamlit as st
import pickle
import numpy as np

# Set page title and styling
st.set_page_config(page_title="Delivery Time Estimator", page_icon="🛵", layout="centered")

st.title("🛵 Food Delivery Time Estimator")
st.write("Enter order details below to estimate delivery time in minutes.")

# 1. Load the model and single unified encoders dictionary
@st.cache_resource
def load_artifacts():
    with open('best_delivery_rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('encoders.pkl', 'rb') as encoders_file:
        encoders = pickle.load(encoders_file)
    return model, encoders

try:
    model, encoders = load_artifacts()
    le_weather = encoders['Weather']
    le_traffic = encoders['Traffic_Level']
except Exception as e:
    st.error(f"Error loading pickle files. Make sure 'best_delivery_rf_model.pkl' and 'encoders.pkl' are in the same folder as app.py. Error: {e}")
    st.stop()

# 2. Set up Input Forms
st.subheader("Order & Route Details")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
    prep_time = st.number_input("Preparation Time (min)", min_value=1, max_value=120, value=15, step=1)
    courier_exp = st.number_input("Courier Experience (years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)

with col2:
    # Use the loaded Encoder classes to fill selection box options dynamically
    weather_option = st.selectbox("Weather Condition", options=list(le_weather.classes_))
    traffic_option = st.selectbox("Traffic Level", options=list(le_traffic.classes_))

# 3. Predict Button
if st.button("⚡ Estimate Delivery Time", use_container_width=True):
    # Transform categorical variables
    encoded_weather = le_weather.transform([weather_option])[0]
    encoded_traffic = le_traffic.transform([traffic_option])[0]
    
    # Arrange input features in the exact training layout:
    # ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 'Weather', 'Traffic_Level']
    input_features = np.array([[distance, prep_time, courier_exp, encoded_weather, encoded_traffic]])
    
    # Predict
    prediction = model.predict(input_features)[0]
    
    # Display Output
    st.success(f"### ⏱️ Estimated Delivery Time: **{prediction:.1f} minutes**")
"""

# Write this code block to a local file inside Colab for the user to easily copy/download
with open('streamlit_app.py', 'w') as f:
    f.write(streamlit_code)
print("A copy of the Streamlit script has also been saved to this runtime as 'streamlit_app.py'")