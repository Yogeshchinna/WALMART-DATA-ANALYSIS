import base64
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


background_image_path = "C:/Users/YOGESH/OneDrive/Desktop/Phase_1/maxresdefault.jpg"
with open(background_image_path, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode()


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{base64_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center center;
}}

[data-testid="stSlider"] > div > div > div > div > div {{
    color: white;  # Set text color to white for sliders
}}

[data-testid="stRadio"] > div {{
    color: white;  # Set text color to white for radio buttons
}}
</style>
"""

# Inject the CSS into the Streamlit page
st.markdown(page_bg_img, unsafe_allow_html=True)  

# Load Data from CSV
data = pd.read_csv("data.csv")


required_columns = ['Store', 'Dept', 'Holiday', 'Weekly_Sales']

if not all(col in data.columns for col in required_columns):
    raise ValueError("Missing required columns in 'data'")

# Create 'new_data' with Expected Columns
new_data = data[required_columns].copy()

# Train Random Forest Model
x = new_data[['Store', 'Dept', 'Holiday']]
y = new_data['Weekly_Sales']

model = RandomForestRegressor()
model.fit(x, y)

# Function to create the Streamlit page
def show_predict_page(data):
    st.title("#Walmart Sales Analysis Prediction")

    # Correlation Heatmap with a Specific Figure
    figure = plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True)
    st.pyplot(figure)

    # Bar Plot for Store vs. Weekly Sales
    figure = plt.figure(figsize=(10, 6))
    sns.barplot(x='Store', y='Weekly_Sales', data=data)
    st.pyplot(figure)

    # Input Controls for Prediction with White Text Color
    store_id = st.slider("Store ID:", 1, 45, 1)
    department = st.slider("Department:", 1, 100, 1)
    holiday = st.radio("Holiday", (True, False))

    # Button for Calculating Weekly Sales
    if st.button("Calculate Weekly Sales"):
        prediction_input = np.array([[store_id, department, holiday]])
        prediction_input.astype(float)
        sales_prediction = model.predict(prediction_input)
        st.subheader(f'Estimated Weekly Sales: {sales_prediction[0]:.2f}')

# Call 'show_predict_page' with 'new_data'
show_predict_page(new_data)
