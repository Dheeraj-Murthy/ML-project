import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import joblib
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Add the parent directory of PreProcessing to the path for module import
sys.path.append(os.path.join(os.path.dirname(__file__), '../PreProcessing/'))

from preprocessing import *

def run_preprocessing(
        train_file="train.csv",
        test_file="test.csv",
        target_log_transform=True,
        poly_degree=2):
    """
    Loads data, performs outlier removal, feature engineering, polynomial feature generation,
    and applies the preprocessor.
    """
    DATA_PATH = os.path.join(
        os.path.dirname(__file__),
        '..',
        'dataset') + os.sep
    PREPROCESSOR_PATH = os.path.join(
        os.path.dirname(__file__),
        '..',
        'fitted_preprocessor.joblib')

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

    print("\nStarting preprocessing steps for Polynomial Regression...")
    X_full_clean, y_full_clean = remove_outliers(X_full, y_full)

    X_full_fe = major_feature_engineering(X_full_clean)
    X_test_fe = major_feature_engineering(X_test)

    y_train_log = np.log1p(
        y_full_clean) if target_log_transform else y_full_clean

    preprocessor = None
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"Successfully loaded preprocessor from {PREPROCESSOR_PATH}.")
    except FileNotFoundError:
        print(
            f"WARNING: Preprocessor not found at {PREPROCESSOR_PATH}. Fitting and saving it now...")
        preprocessor = create_and_fit_preprocessor(X_full_fe)

    X_train_proc = preprocessor.transform(X_full_fe)
    X_test_proc = preprocessor.transform(X_test_fe)

    # --- Polynomial Features ---
    print(f"Generating polynomial features with degree {poly_degree}...")
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
    X_train_poly = poly.fit_transform(X_train_proc)
    X_test_poly = poly.transform(X_test_proc)

    print(
        f"Data ready. Training data shape: {X_train_poly.shape}, Test data shape: {X_test_poly.shape}")

    return X_train_poly, X_test_poly, y_train_log, test_ids, preprocessor

def run_training_and_prediction():
    """
    Executes the Polynomial Regression model training, evaluation, and submission creation.
    """
    X_full_train_poly, X_test_poly, y_full_train_log, test_ids, preprocessor = run_preprocessing(
        train_file="train.csv", test_file="test.csv", target_log_transform=True, poly_degree=2)

    X_sub_train, X_val, y_sub_train_log, y_val_log = train_test_split(
        X_full_train_poly, y_full_train_log, test_size=0.2, random_state=42
    )

    poly_model = LinearRegression(n_jobs=-1)

    print("\n--- Model Training & Evaluation (Polynomial Regression) ---")

    poly_model.fit(X_sub_train, y_sub_train_log)

    val_log_predictions = poly_model.predict(X_val)
    val_r2 = r2_score(y_val_log, val_log_predictions)
    val_mse = mean_squared_error(y_val_log, val_log_predictions)

    print(f"Validation R-squared (Log-transformed): {val_r2:.4f}")
    print(f"Validation Mean Squared Error (Log-transformed): {val_mse:.4f}")

    val_predictions_orig = np.expm1(val_log_predictions)
    y_val_orig = np.expm1(y_val_log)

    val_rmse_orig = np.sqrt(mean_squared_error(y_val_orig, val_predictions_orig))

    print(f"Validation RMSE (Original Scale): {val_rmse_orig:.4f}")

    save_model_results(
        os.path.basename(__file__),
        'PolynomialRegression',
        val_rmse_orig)

    print("\nFitting final production model on ALL training data...")
    poly_model.fit(X_full_train_poly, y_full_train_log)

    model_filename = 'fitted_poly_model.joblib'
    joblib.dump(poly_model, model_filename)

    print(
        f"Final trained Polynomial Regression model saved to '{model_filename}'.")

    submission_filename = 'submission_poly.csv'

    print(f"\nGenerating predictions using the final production model...")

    test_log_predictions = poly_model.predict(X_test_poly)
    test_predictions = np.expm1(test_log_predictions)

    submission_df = pd.DataFrame(
        {'Id': test_ids, 'HotelValue': test_predictions})
    submission_df.to_csv(submission_filename, index=False)

    print(f"\nSubmission file '{submission_filename}' created successfully.")
    print("\nFirst 5 predictions in the submission file:")
    print(submission_df.head())

if __name__ == '__main__':
    run_training_and_prediction()
