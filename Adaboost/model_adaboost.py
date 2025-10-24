import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor # Switched to AdaBoost Regressor
from sklearn.tree import DecisionTreeRegressor # Base estimator for AdaBoost
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
# Import the run_preprocessing function from the preprocessing file
from preprocessing import run_preprocessing, major_feature_engineering, create_preprocessor_pipeline

# --- 1. Load and Prepare Data ---

# Run the complete preprocessing pipeline to get the processed data arrays
# We use target_log_transform=True, which is CRITICAL for tree-based models on skewed price data.
X_full_train_proc, X_test_proc, y_full_train_log, test_ids, preprocessor = run_preprocessing(
    train_file="train.csv", 
    test_file="test.csv", 
    target_log_transform=True
)

# Create a Train-Validation Split for metric reporting (80/20 split on processed data)
X_sub_train, X_val, y_sub_train_log, y_val_log = train_test_split(
    X_full_train_proc, y_full_train_log, test_size=0.2, random_state=42
)

# --- 2. Define the Tree-Based Model (AdaBoost Regressor) ---

# AdaBoost sequentially focuses on difficult-to-predict samples.
# We use a shallow DecisionTree as the base estimator (weak learner) for quick training.
base_estimator = DecisionTreeRegressor(max_depth=4)

ada_model = AdaBoostRegressor(
    estimator=base_estimator,
    n_estimators=100, # Number of weak learners
    learning_rate=0.1, # Weight applied to each estimator at each boosting iteration
    random_state=42
) 

# --- 3. Train Model on Full Data and Report Metrics ---

print("\n--- Model Training & Evaluation (AdaBoost Regressor) ---")

# Train the model on the sub-training split for evaluation
ada_model.fit(X_sub_train, y_sub_train_log)

# Calculate metrics (using log-transformed values)
val_log_predictions = ada_model.predict(X_val)
val_r2 = r2_score(y_val_log, val_log_predictions)
val_mse = mean_squared_error(y_val_log, val_log_predictions)

print(f"Validation R-squared (Log-transformed): {val_r2:.4f}")
print(f"Validation Mean Squared Error (Log-transformed): {val_mse:.4f}")

# --- 4. Create Final Production Model and Save ---

print("\nFitting final production model on ALL training data...")

# Re-train the AdaBoost model on the FULL processed data
ada_model.fit(X_full_train_proc, y_full_train_log)

# Save the trained model object for the prediction script
joblib.dump(ada_model, 'fitted_adaboost_model.joblib')

print("Final trained AdaBoost model saved to 'fitted_adaboost_model.joblib'.")

# Note: The fitted_preprocessor.joblib was saved by run_preprocessing().
