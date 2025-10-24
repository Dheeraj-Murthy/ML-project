import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Standard Linear Regression (Non-Regularized)
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
# Import the run_preprocessing function from the preprocessing file
from preprocessing import run_preprocessing # This function handles loading/FE/OHE/Scaling/Outlier Removal

# --- 1. Load and Prepare Data ---

# Run the complete preprocessing pipeline to get the processed data arrays
# We use target_log_transform=True, which is CRITICAL for linear models on skewed price data.
X_full_train_proc, X_test_proc, y_full_train_log, test_ids, preprocessor = run_preprocessing(
    train_file="train.csv", 
    test_file="test.csv", 
    target_log_transform=True
)

# Create a Train-Validation Split for metric reporting (80/20 split on processed data)
X_sub_train, X_val, y_sub_train_log, y_val_log = train_test_split(
    X_full_train_proc, y_full_train_log, test_size=0.2, random_state=42
)

# --- 2. Define the Linear Model (Standard Linear Regression) ---

# LinearRegression performs standard Ordinary Least Squares (OLS) without regularization.
# Note: This is highly susceptible to multicollinearity from One-Hot Encoding.
linear_model = LinearRegression(n_jobs=-1) 

# --- 3. Train Model on Full Data and Report Metrics ---

print("\n--- Model Training & Evaluation (Standard Linear Regression / OLS) ---")

# Train the model on the sub-training split for evaluation
linear_model.fit(X_sub_train, y_sub_train_log)

# Calculate metrics (using log-transformed values)
val_log_predictions = linear_model.predict(X_val)
val_r2 = r2_score(y_val_log, val_log_predictions)
val_mse = mean_squared_error(y_val_log, val_log_predictions)

print(f"Validation R-squared (Log-transformed): {val_r2:.4f}")
print(f"Validation Mean Squared Error (Log-transformed): {val_mse:.4f}")

# --- 4. Create Final Production Model and Save ---

print("\nFitting final production model on ALL training data...")

# Re-train the linear model on the FULL processed data
linear_model.fit(X_full_train_proc, y_full_train_log)

# Save the trained model object 
model_filename = 'fitted_linear_model.joblib'
joblib.dump(linear_model, model_filename)

print(f"Final trained Linear Regression model saved to '{model_filename}'.")

# --- 5. Prediction and Submission ---

submission_filename = 'submission_linear_regression.csv'

print(f"\nGenerating predictions using the final production model...")

# Predict on the processed test data (log values)
test_log_predictions = linear_model.predict(X_test_proc)

# Reverse the Log-transformation: HotelValue = exp(log_predictions) - 1
test_predictions = np.expm1(test_log_predictions)

# Create the submission file
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file '{submission_filename}' created successfully.")
print("\nFirst 5 predictions in the submission file:")
print(submission_df.head())
