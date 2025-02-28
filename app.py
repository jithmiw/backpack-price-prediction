import numpy as np
import pandas as pd
import pickle
import logging
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load trained models
with open("models/classifiers/random_forest_v1.pkl", "rb") as rf_file:
    rf = pickle.load(rf_file)
with open("models/classifiers/xgboost_v1.pkl", "rb") as xgb_file:
    xgb = pickle.load(xgb_file)
meta_model = pickle.load(open("models/meta_models/meta_model_v1.pkl", "rb"))
ann = load_model("models/neural_networks/neural_network_v1.keras", compile=False)        

# Load preprocessing objects
with open("models/encoders/encoder_v1.pkl", "rb") as enc_file:
    encoder = pickle.load(enc_file)
with open("models/scalers/scaler_v1.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        input_data = request.form.to_dict()
        
        # Validate numeric fields
        try:
            input_data['Compartments'] = float(input_data['Compartments'])
            input_data['Weight Capacity (kg)'] = float(input_data['Weight Capacity (kg)'])
        except ValueError:
            return jsonify({"error": "Invalid numeric input. Please enter valid numbers."}), 400
        
        # Convert categorical features using LabelEncoder
        categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
        for col in categorical_cols:
            if input_data[col] in encoder[col].classes_:
                input_data[col] = encoder[col].transform([input_data[col]])[0]
            else:
                input_data[col] = -1  # Handle unknown categories
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Standardize numerical columns
        numerical_cols = ['Compartments', 'Weight Capacity (kg)']
        df[numerical_cols] = scaler.transform(df[numerical_cols])

        # Ensure correct column order
        df = df[rf.feature_names_in_]
        
        # Predict using models
        rf_pred = rf.predict(df)[0]
        xgb_pred = xgb.predict(df)[0]
        nn_pred = ann.predict(df)[0][0] 
        
        # Meta-model prediction
        meta_pred = meta_model.predict(np.array([[rf_pred, xgb_pred, nn_pred]]))[0]
        
        logging.info("Prediction successful")
        return jsonify({
            'Random Forest Prediction': float(rf_pred),
            'XGBoost Prediction': float(xgb_pred),
            'Neural Network Prediction': float(nn_pred),
            'Final Price Prediction': float(meta_pred)
        })
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An error occurred during prediction. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True)