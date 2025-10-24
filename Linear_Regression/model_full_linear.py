import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Standard Linear Regression (Non-Regularized)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import sys
import warnings
# Necessary imports for the custom class
import numpy as np


warnings.filterwarnings('ignore')

# Custom Ordinal Encoder Class (COPIED from preprocessing.py) ---
# --- 2. Custom Ordinal Encoder Class (Full Implementation Required) ---


class OrdinalEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, mappings):
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.mappings.items():
            # Apply the actual mapping logic
            X_copy[col] = X_copy[col].fillna('None').astype(
                str).map(mapping).fillna(0).astype(int)
        return X_copy


QUAL_MAPPING = {
    'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, 'NA': 0, np.nan: 0
}
BSMT_HEIGHT_MAPPING = {
    'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, np.nan: 0
}

# Add the parent directory of PreProcessing to the path for module import
sys.path.append(os.path.join(os.path.dirname(__file__), '../PreProcessing'))

from preprocessing import major_feature_engineering, create_and_fit_preprocessor, remove_outliers, save_model_results

# Import custom functions from the preprocessing module
# NOTE: The custom functions are used below inside run_preprocessing


def run_preprocessing(
        train_file="train.csv",
        test_file="test.csv",
        target_log_transform=True):
    """
    Loads data, performs outlier removal and feature engineering, and applies
    (or fits/saves, if missing) the ColumnTransformer using the functions
    defined in the imported preprocessing module.

    Returns processed training/test data, log-transformed target, test IDs, and the preprocessor.
    """
    # Define paths relative to the current script's location
    DATA_PATH = os.path.join(
        os.path.dirname(__file__),
        '..',
        'dataset') + os.sep
    PREPROCESSOR_PATH = os.path.join(
        os.path.dirname(__file__),
        '..',
        'PreProcessing',
        'fitted_preprocessor.joblib')

    # --- 1. Load Data ---
    try:
        train_df = pd.read_csv(DATA_PATH + train_file)
        test_df = pd.read_csv(DATA_PATH + test_file)
    except FileNotFoundError as e:
        print(
            f"ERROR: Could not find data files. Ensure they are in the '{DATA_PATH}' folder.")
        raise e

    test_ids = test_df['Id']
    y_full = train_df['HotelValue']
    X_full = train_df.drop(columns=['Id', 'HotelValue'])
    X_test = test_df.drop(columns=['Id'])

    # --- 2. Outlier Removal (Applied only to training data) ---
    print("\nStarting preprocessing steps...")
    X_full_clean, y_full_clean = remove_outliers(X_full, y_full)

    # --- 3. Feature Engineering (Applied to clean train and raw test data) ---
    X_full_fe = major_feature_engineering(X_full_clean)
    X_test_fe = major_feature_engineering(X_test)

    # --- 4. Target Transformation ---
    y_train_log = np.log1p(
        y_full_clean) if target_log_transform else y_full_clean

    # --- 5. Load/Fit Preprocessor ---
    preprocessor = None
    try:
        # Load the preprocessor (assumes it was fitted by a separate script
        # run)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"Successfully loaded preprocessor from {PREPROCESSOR_PATH}.")
    except FileNotFoundError:
        print(
            f"WARNING: Preprocessor not found at {PREPROCESSOR_PATH}. Fitting and saving it now...")
        # If not found, fit and save it using the function from
        # preprocessing.py
        preprocessor = create_and_fit_preprocessor(X_full_fe)

    # --- 6. Transform Data ---
    # Transform to NumPy arrays as Linear Regression works optimally with them
    # FIX: Removed .values as preprocessor.transform returns a NumPy array
    # when sparse_output=False
    X_train_proc = preprocessor.transform(X_full_fe)
    X_test_proc = preprocessor.transform(X_test_fe)

    print(
        f"Data ready. Training data shape: {
            X_train_proc.shape}, Test data shape: {
            X_test_proc.shape}")

    return X_train_proc, X_test_proc, y_train_log, test_ids, preprocessor


def run_training_and_prediction():
    """
    Executes the Linear Regression model training, evaluation, and submission creation.
    """
    # --- 1. Load and Prepare Data ---
    # Run the complete preprocessing pipeline to get the processed data arrays
    X_full_train_proc, X_test_proc, y_full_train_log, test_ids, preprocessor = run_preprocessing(
        train_file="train.csv", test_file="test.csv", target_log_transform=True)

    # Create a Train-Validation Split for metric reporting (80/20 split on
    # processed data)
    X_sub_train, X_val, y_sub_train_log, y_val_log = train_test_split(
        X_full_train_proc, y_full_train_log, test_size=0.2, random_state=42
    )

    # --- 2. Define the Linear Model (Standard Linear Regression) ---

    # LinearRegression performs standard Ordinary Least Squares (OLS) without
    # regularization.
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

    # --- METRIC FIX: Calculate RMSE on the original scale for proper reporting ---
    # Inverse transform both predicted and actual values to their original scale
    val_predictions_orig = np.expm1(val_log_predictions)
    y_val_orig = np.expm1(y_val_log)

    # Calculate RMSE on the original, non-log scale
    val_rmse_orig = np.sqrt(mean_squared_error(y_val_orig, val_predictions_orig))

    print(f"Validation RMSE (Original Scale): {val_rmse_orig:.4f}")

    # Save the original-scale RMSE to the results file
    save_model_results(
        os.path.basename(__file__),
        'LinearRegression',
        val_rmse_orig)

    # --- 4. Create Final Production Model and Save ---

    print("\nFitting final production model on ALL training data...")

    # Re-train the linear model on the FULL processed data
    linear_model.fit(X_full_train_proc, y_full_train_log)

    # Save the trained model object
    model_filename = 'fitted_linear_model.joblib'
    joblib.dump(linear_model, model_filename)

    print(
        f"Final trained Linear Regression model saved to '{model_filename}'.")

    # --- 5. Prediction and Submission ---

    submission_filename = 'submission_linear_regression.csv'

    print(f"\nGenerating predictions using the final production model...")

    # Predict on the processed test data (log values)
    test_log_predictions = linear_model.predict(X_test_proc)

    # Reverse the Log-transformation: HotelValue = exp(log_predictions) - 1
    test_predictions = np.expm1(test_log_predictions)

    # Create the submission file
    submission_df = pd.DataFrame(
        {'Id': test_ids, 'HotelValue': test_predictions})
    submission_df.to_csv(submission_filename, index=False)

    print(f"\nSubmission file '{submission_filename}' created successfully.")
    print("\nFirst 5 predictions in the submission file:")
    print(submission_df.head())


if __name__ == '__main__':
    run_training_and_prediction()
