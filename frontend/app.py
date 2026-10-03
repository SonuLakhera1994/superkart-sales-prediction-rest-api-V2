
import os

import pandas as pd
import requests
import streamlit as st

# Base URL of the Flask backend. Defaults to localhost for local development.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860")

# Set the title of the Streamlit app
st.title("SuperKart product sales prediction")

# Section for online prediction
st.subheader("Online Prediction")

product_weight = st.number_input("Product Weight", min_value=0.0, value=1.0)
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, value=1.0)
product_mrp = st.number_input("Product MRP", min_value=0.0, value=1.0)
store_age_years = st.number_input("Store Age Years", min_value=0, value=1)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Medium Sugar", "High Sugar"])
store_size = st.selectbox("Store Size", ["Small", "Medium", "Large"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
product_id_char = st.selectbox("Product ID Character", ["FD", "FW"])
product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Allocated_Area": product_allocated_area,
    "Product_MRP": product_mrp,
    "Store_Age_Years": store_age_years,
    "Product_Sugar_Content": product_sugar_content,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_location_city_type,
    "Store_Type": store_type,
    "Product_Id_char": product_id_char,
    "Product_Type_Category": product_type_category
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_data.to_dict(orient='records')[0], timeout=20)
        if response.status_code == 200:
            prediction = response.json().get("Predicted Sales")
            st.success(f"Predicted Sales: {prediction}")
        else:
            st.error(f"Unable to connect to the prediction API. Status: {response.status_code}")
    except requests.RequestException as exc:
        st.error(f"Unable to connect to the prediction API: {exc}")

# Section for batch prediction
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file}, timeout=20)
            if response.status_code == 200:
                predictions = response.json()
                st.success("Batch predictions completed!")
                st.write(predictions)
            else:
                st.error(f"Unable to connect to the prediction API. Status: {response.status_code}")
        except requests.RequestException as exc:
            st.error(f"Unable to connect to the prediction API: {exc}")
