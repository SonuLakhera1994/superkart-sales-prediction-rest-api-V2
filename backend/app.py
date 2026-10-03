
# Import necessary libraries
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# /content/deployment_files/super_kart_prediction_model_v2_0.joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "super_kart_prediction_model_v2_0.joblib"

FEATURE_ALIASES = {
    "Product_Weight": ["Product_Weight", "product_weight"],
    "Product_Allocated_Area": ["Product_Allocated_Area", "product_allocated_area"],
    "Product_MRP": ["Product_MRP", "product_mrp"],
    "Store_Age_Years": ["Store_Age_Years", "store_age_years"],
    "Product_Sugar_Content": ["Product_Sugar_Content", "product_sugar_content"],
    "Store_Size": ["Store_Size", "store_size"],
    "Store_Location_City_Type": ["Store_Location_City_Type", "store_location_city_type"],
    "Store_Type": ["Store_Type", "store_type"],
    "Product_Id_char": ["Product_Id_char", "product_id_char"],
    "Product_Type_Category": ["Product_Type_Category", "product_type_category"],
}

# Initialize the Flask application
product_Store_Sales_api = Flask("SuperKart product store sales Predictor")

# Load the trained machine learning model
model = joblib.load(MODEL_PATH)


def extract_feature_row(payload):
    raw = payload or {}
    sample = {}
    for canonical, aliases in FEATURE_ALIASES.items():
        for key in aliases:
            if key in raw:
                sample[canonical] = raw[key]
                break
        else:
            raise KeyError(f"Missing required feature: {canonical}")
    return pd.DataFrame([sample])


# Define a route for the home page
@product_Store_Sales_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"


@product_Store_Sales_api.get('/health')
def health():
    return jsonify({"status": "ok"})


# Define an endpoint for single prediction
@product_Store_Sales_api.post('/v1/predict')
def predict_sales():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must contain JSON features."}), 400

    try:
        input_data = extract_feature_row(payload)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400

    predicted_sales = model.predict(input_data)[0]
    predicted_sales = round(float(predicted_sales), 2)

    return jsonify({"Predicted Sales": predicted_sales})


# Define an endpoint for batch prediction
@product_Store_Sales_api.post('/v1/predictbatch')
def predict_sales_batch():
    if 'file' not in request.files:
        return jsonify({"error": "CSV file is required for batch prediction."}), 400

    uploaded_file = request.files['file']
    input_data = pd.read_csv(uploaded_file)

    predicted_sales = model.predict(input_data)
    predicted_sales = [round(float(prediction), 2) for prediction in predicted_sales]

    if 'id' in input_data.columns:
        property_ids = input_data['id'].astype(str).tolist()
        output_dict = dict(zip(property_ids, predicted_sales))
    else:
        output_dict = {str(index): value for index, value in enumerate(predicted_sales)}

    return jsonify(output_dict)


# Run Flask application
if __name__ == '__main__':
    product_Store_Sales_api.run(host='0.0.0.0', port=7860, debug=False)
