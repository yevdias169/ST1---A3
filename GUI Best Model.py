import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import tkinter as tk
from tkinter import ttk
import os

# 1. Train and save the model
def train_and_save_model():
    try:
        # Load the data
        df = pd.read_csv('Medical_insurance_encoded.csv')
        
        # Separate features and target
        X = df.drop('charges', axis=1)
        y = df['charges']
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Save the model, scaler, and feature names
        joblib.dump(model, 'medical_insurance_model.joblib')
        joblib.dump(scaler, 'medical_insurance_scaler.joblib')
        joblib.dump(X.columns.tolist(), 'feature_names.joblib')
        
        print("Model, scaler, and feature names have been trained and saved.")
    except Exception as e:
        print(f"An error occurred while training the model: {e}")
        raise

# 2. Load the saved model, scaler, and feature names
def load_model_and_scaler():
    try:
        model = joblib.load('medical_insurance_model.joblib')
        scaler = joblib.load('medical_insurance_scaler.joblib')
        feature_names = joblib.load('feature_names.joblib')
        return model, scaler, feature_names
    except FileNotFoundError:
        print("Model, scaler, or feature names file not found. Please train the model first.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise

# 3. Create a Python function for prediction
def predict_charges(age, sex, bmi, children, smoker, region):
    model, scaler, feature_names = load_model_and_scaler()
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children]
    })
    
    # Encode categorical variables
    input_data[f'sex_{sex}'] = 1
    input_data[f'smoker_{smoker}'] = 1
    input_data[f'region_{region}'] = 1
    
    # Ensure all columns from training are present
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0
    
    # Reorder columns to match training data
    input_data = input_data[feature_names]
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    return prediction[0]

# 4. Test the function with new unseen data
def test_prediction_function():
    test_data = {
        'age': 35,
        'sex': 'male',
        'bmi': 27.5,
        'children': 2,
        'smoker': 'yes',
        'region': 'northeast'
    }
    
    try:
        prediction = predict_charges(**test_data)
        print(f"Predicted charges for test data: ${prediction:.2f}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

# 5. Implement as a desktop application using tkinter
def create_gui():
    def on_predict():
        try:
            age = int(age_entry.get())
            sex = sex_var.get()
            bmi = float(bmi_entry.get())
            children = int(children_entry.get())
            smoker = smoker_var.get()
            region = region_var.get()
            
            if not all([age, sex, bmi, children, smoker, region]):
                raise ValueError("All fields must be filled.")
            
            if age <= 0:
                raise ValueError("Age must be a positive integer.")
            
            if bmi <= 0:
                raise ValueError("BMI must be a positive number.")
            
            if children < 0:
                raise ValueError("Number of children must be a non-negative integer.")
            
            prediction = predict_charges(age, sex, bmi, children, smoker, region)
            result_label.config(text=f"Predicted Insurance Charges: ${prediction:.2f}")
        except ValueError as ve:
            result_label.config(text=f"Input Error: {str(ve)}")
        except Exception as e:
            result_label.config(text=f"An error occurred: {str(e)}")

    root = tk.Tk()
    root.title("Medical Insurance Cost Predictor")

    # Create and place widgets
    ttk.Label(root, text="Age:").grid(row=0, column=0, padx=5, pady=5)
    age_entry = ttk.Entry(root)
    age_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(root, text="Sex:").grid(row=1, column=0, padx=5, pady=5)
    sex_var = tk.StringVar()
    ttk.Combobox(root, textvariable=sex_var, values=['male', 'female']).grid(row=1, column=1, padx=5, pady=5)

    ttk.Label(root, text="BMI:").grid(row=2, column=0, padx=5, pady=5)
    bmi_entry = ttk.Entry(root)
    bmi_entry.grid(row=2, column=1, padx=5, pady=5)

    ttk.Label(root, text="Children:").grid(row=3, column=0, padx=5, pady=5)
    children_entry = ttk.Entry(root)
    children_entry.grid(row=3, column=1, padx=5, pady=5)

    ttk.Label(root, text="Smoker:").grid(row=4, column=0, padx=5, pady=5)
    smoker_var = tk.StringVar()
    ttk.Combobox(root, textvariable=smoker_var, values=['yes', 'no']).grid(row=4, column=1, padx=5, pady=5)

    ttk.Label(root, text="Region:").grid(row=5, column=0, padx=5, pady=5)
    region_var = tk.StringVar()
    ttk.Combobox(root, textvariable=region_var, values=['northeast', 'northwest', 'southeast', 'southwest']).grid(row=5, column=1, padx=5, pady=5)

    ttk.Button(root, text="Predict", command=on_predict).grid(row=6, column=0, columnspan=2, pady=10)

    result_label = ttk.Label(root, text="")
    result_label.grid(row=7, column=0, columnspan=2, pady=5)

    root.mainloop()

if __name__ == "__main__":
    # Check if the model file exists, if not, train and save the model
    if not os.path.exists('medical_insurance_model.joblib') or \
       not os.path.exists('medical_insurance_scaler.joblib') or \
       not os.path.exists('feature_names.joblib'):
        print("Model, scaler, or feature names file not found. Training a new model...")
        train_and_save_model()
    
    # Test the prediction function
    test_prediction_function()
    
    # Launch the GUI
    create_gui()