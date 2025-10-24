import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso # Switched to Lasso Regression (L1 Regularization)
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
# Import the run_preprocessing function from the preprocessing file
from preprocessing import run_preprocessing, major_feature_engineering, create_preprocessor_pipeline

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

# --- 2. Define the Linear Model (Lasso Optimization) ---

# Lasso performs L1 regularization, which shrinks less important feature coefficients to zero (feature selection).
# Alpha (0.001) is the regularization strength. Max_iter is increased for guaranteed convergence on complex OHE data.
lasso_model = Lasso(alpha=0.001, max_iter=5000, random_state=42) 

# --- 3. Train Model on Full Data and Report Metrics ---

print("\n--- Model Training & Evaluation (Lasso Optimization / L1 Regularization) ---")

# Train the model on the sub-training split for evaluation
lasso_model.fit(X_sub_train, y_sub_train_log)

# Calculate metrics (using log-transformed values)
val_log_predictions = lasso_model.predict(X_val)
val_r2 = r2_score(y_val_log, val_log_predictions)
val_mse = mean_squared_error(y_val_log, val_log_predictions)

print(f"Validation R-squared (Log-transformed): {val_r2:.4f}")
print(f"Validation Mean Squared Error (Log-transformed): {val_mse:.4f}")

# --- 4. Create Final Production Model and Save ---

print("\nFitting final production model on ALL training data...")

# Re-train the Lasso model on the FULL processed data
lasso_model.fit(X_full_train_proc, y_full_train_log)

# Save the trained model object for the prediction script
joblib.dump(lasso_model, 'fitted_lasso_model.joblib')

print("Final trained Lasso model saved to 'fitted_lasso_model.joblib'.")

# Note: The fitted_preprocessor.joblib was saved by run_preprocessing().
