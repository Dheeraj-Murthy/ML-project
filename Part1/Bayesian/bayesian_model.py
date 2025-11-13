import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import sys
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '../PreProcessing'))

from preprocessing import *


def run_model_training_and_prediction(
        train_file="train.csv",
        test_file="test.csv",
        target_log_transform=True):
    """
    Executes the full pipeline: data loading, preprocessing,
    Bayesian Ridge training, evaluation, and submission file creation.
    """
    try:
        # --- 1. Load and Prepare Data ---
        X_full_train_proc, X_test_proc, y_full_train_log, y_full_clean, test_ids, preprocessor = run_preprocessing(
            train_file=train_file,
            test_file=test_file,
            target_log_transform=target_log_transform
        )
    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        return

    # Create a Train-Validation Split for metric reporting (80/20 split)
    X_sub_train, X_val, y_sub_train_log, y_val_log = train_test_split(
        X_full_train_proc, y_full_train_log, test_size=0.2, random_state=42
    )

    # --- 2. Define and Train the Model (Bayesian Ridge) ---
    bayesian_model = BayesianRidge()

    print("\n--- Model Training & Evaluation (Bayesian Ridge Regression) ---")

    # Train the model on the sub-training split for evaluation
    bayesian_model.fit(X_sub_train, y_sub_train_log)

    # Calculate metrics (using log-transformed values)
    val_log_predictions = bayesian_model.predict(X_val)
    val_r2 = r2_score(y_val_log, val_log_predictions)
    val_mse = mean_squared_error(y_val_log, val_log_predictions)

    print(f"Validation R-squared (Log-transformed): {val_r2:.4f}")
    print(f"Validation Mean Squared Error (Log-transformed): {val_mse:.4f}")

    # Calculate RMSE on original scale
    # We need to split y_full_clean in the same way the data was split to get the validation set
    _, y_val_original = train_test_split(y_full_clean, test_size=0.2, random_state=42)
    val_predictions_original = np.expm1(val_log_predictions)
    rmse_original = np.sqrt(mean_squared_error(y_val_original, val_predictions_original))
    print(f"Validation RMSE (Original Scale): {rmse_original:.4f}")
    save_model_results(
        os.path.basename(__file__),
        'BayesianRidge',
        rmse_original)

    # --- 3. Create Final Production Model and Save ---

    print("\nFitting final production model on ALL training data...")

    # Re-train the Bayesian model on the FULL processed data
    bayesian_model.fit(X_full_train_proc, y_full_train_log)

    # Save the trained model object (saved in the Bayesian folder)
    joblib.dump(bayesian_model, 'fitted_bayesian_model.joblib')

    print("Final trained Bayesian Ridge model saved to 'fitted_bayesian_model.joblib'.")

    # --- 4. Prediction and Submission ---

    # Predict on the actual test data (log values)
    test_log_predictions = bayesian_model.predict(X_test_proc)

    # Reverse the Log-transformation
    test_predictions = np.expm1(test_log_predictions)

    # Create the submission file
    submission_filename = 'submission_bayesian.csv'
    submission_df = pd.DataFrame(
        {'Id': test_ids, 'HotelValue': test_predictions})
    submission_df.to_csv(submission_filename, index=False)

    print(
        f"\nSubmission file '{submission_filename}' created successfully in the Bayesian folder.")


if __name__ == '__main__':
    # Execute the full pipeline
    run_model_training_and_prediction(target_log_transform=True)
