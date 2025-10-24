import pandas as pd
import numpy as np
import joblib
import os
# The major_feature_engineering function is needed to transform the raw test data
from preprocessing import major_feature_engineering 

def run_prediction(test_file="test.csv", 
                   preprocessor_file="fitted_preprocessor.joblib", 
                   # This parameter is set via the call in __main__
                   model_file="fitted_xgboost_model.joblib", 
                   submission_filename="submission_xgboost.csv"):
    """
    Loads the test data, applies saved preprocessing steps, generates predictions
    using the trained model, and saves the final submission file.
    """
    
    print("--- Starting Prediction Process ---")
    
    # --- 1. Load Assets ---
    try:
        # Load test data
        test_df = pd.read_csv(test_file)
        test_ids = test_df['Id']
        X_test = test_df.drop('Id', axis=1)

        # Load the fitted preprocessor
        preprocessor = joblib.load(preprocessor_file)
        
        # Load the trained model
        model = joblib.load(model_file)
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Make sure {e.filename} exists.")
        return

    # --- 2. Apply Preprocessing and Feature Engineering ---
    
    # Apply the exact same feature engineering used during training
    X_test_fe = major_feature_engineering(X_test)
    
    # Transform the feature-engineered test data using the saved preprocessor
    X_test_processed = preprocessor.transform(X_test_fe)
    
    print(f"Test data transformed successfully. Shape: {X_test_processed.shape}")

    # --- 3. Generate Predictions ---
    
    # Predict on the processed test data (output will be log-transformed values)
    log_predictions = model.predict(X_test_processed)
    
    # Reverse the Log-transformation: HotelValue = exp(log_predictions) - 1
    # This converts the predictions back to the original dollar scale.
    final_predictions = np.expm1(log_predictions)
    
    # Ensure predictions are non-negative 
    final_predictions[final_predictions < 0] = 0
    
    print("Predictions generated and inverse-transformed.")

    # --- 4. Create Submission File ---
    
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'HotelValue': final_predictions
    })
    
    submission_df.to_csv(submission_filename, index=False)
    
    print(f"\nSubmission file '{submission_filename}' created successfully.")
    print("\nFirst 5 predictions:")
    print(submission_df.head())

if __name__ == '__main__':
    # Change the model_file argument to load the desired model:
    # Example is set to Ridge Regression, assuming that model was trained last.
    run_prediction(
        model_file="fitted_linear_model.joblib",
        submission_filename="submission_linear.csv"
    )
