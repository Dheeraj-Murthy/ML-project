import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Switched to Random Forest Regressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Add the parent directory of PreProcessing to the path for module import
sys.path.append(os.path.join(os.path.dirname(__file__), '../PreProcessing'))

# Import custom functions from the preprocessing module
# NOTE: The custom functions are used below inside run_preprocessing
from preprocessing import major_feature_engineering, create_and_fit_preprocessor, remove_outliers 


def run_preprocessing(train_file="train.csv", test_file="test.csv", target_log_transform=True):
    """
    Loads data, performs outlier removal and feature engineering, and applies
    (or fits/saves, if missing) the ColumnTransformer using the functions
    defined in the imported preprocessing module.
    
    Returns processed training/test data, log-transformed target, test IDs, and the preprocessor.
    """
    # Define paths relative to the current script's location
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset') + os.sep
    PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'PreProcessing', 'fitted_preprocessor.joblib')

    # --- 1. Load Data ---
    try:
        train_df = pd.read_csv(DATA_PATH + train_file)
        test_df = pd.read_csv(DATA_PATH + test_file)
    except FileNotFoundError as e:
        print(f"ERROR: Could not find data files. Ensure they are in the '{DATA_PATH}' folder.")
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
    y_train_log = np.log1p(y_full_clean) if target_log_transform else y_full_clean

    # --- 5. Load/Fit Preprocessor ---
    preprocessor = None
    try:
        # Load the preprocessor (assumes it was fitted by a separate script run)
        preprocessor = joblib.load(PREPROCESSOR_PATH) 
        print(f"Successfully loaded preprocessor from {PREPROCESSOR_PATH}.")
    except FileNotFoundError:
        print(f"WARNING: Preprocessor not found at {PREPROCESSOR_PATH}. Fitting and saving it now...")
        # If not found, fit and save it using the function from preprocessing.py
        preprocessor = create_and_fit_preprocessor(X_full_fe)
        
    # --- 6. Transform Data ---
    # Transform to NumPy arrays as tree-based models work optimally with them
    X_train_proc = preprocessor.transform(X_full_fe)
    X_test_proc = preprocessor.transform(X_test_fe)
    
    print(f"Data ready. Training data shape: {X_train_proc.shape}, Test data shape: {X_test_proc.shape}")
    
    return X_train_proc, X_test_proc, y_train_log, test_ids, preprocessor


def run_training_and_prediction():
    """
    Executes the Random Forest model training, evaluation, and submission creation.
    """
    # --- 1. Load and Prepare Data ---
    # Run the complete preprocessing pipeline to get the processed data arrays
    X_full_train_proc, X_test_proc, y_full_train_log, test_ids, preprocessor = run_preprocessing(
        train_file="train.csv", 
        test_file="test.csv", 
        target_log_transform=True
    )

    # Create a Train-Validation Split for metric reporting (80/20 split on processed data)
    X_sub_train, X_val, y_sub_train_log, y_val_log = train_test_split(
        X_full_train_proc, y_full_train_log, test_size=0.2, random_state=42
    )

    # --- 2. Define the Tree-Based Model (Random Forest) ---

    # Random Forest uses an ensemble of decision trees, averaging their predictions.
    rf_model = RandomForestRegressor(
        n_estimators=1000, 
        max_depth=15, 
        min_samples_leaf=5, 
        random_state=42, 
        n_jobs=-1
    ) 

    # --- 3. Train Model on Split Data and Report Metrics ---

    print("\n--- Model Training & Evaluation (Random Forest Regressor) ---")

    # Train the model on the sub-training split for evaluation
    rf_model.fit(X_sub_train, y_sub_train_log)

    # Calculate metrics (using log-transformed values)
    val_log_predictions = rf_model.predict(X_val)
    val_r2 = r2_score(y_val_log, val_log_predictions)
    val_mse = mean_squared_error(y_val_log, val_log_predictions)

    print(f"Validation R-squared (Log-transformed): {val_r2:.4f}")
    print(f"Validation Mean Squared Error (Log-transformed): {val_mse:.4f}")

    # --- 4. Create Final Production Model and Save ---

    print("\nFitting final production model on ALL training data...")

    # Re-train the Random Forest model on the FULL processed data
    rf_model.fit(X_full_train_proc, y_full_train_log)

    # Save the trained model object 
    model_filename = 'fitted_random_forest_model.joblib'
    joblib.dump(rf_model, model_filename)

    print(f"Final trained Random Forest model saved to '{model_filename}'.")

    # --- 5. Prediction and Submission ---

    submission_filename = 'submission_random_forest.csv'

    print(f"\nGenerating predictions using the final production model...")

    # Predict on the processed test data (log values)
    test_log_predictions = rf_model.predict(X_test_proc)

    # Reverse the Log-transformation: HotelValue = exp(log_predictions) - 1
    test_predictions = np.expm1(test_log_predictions)

    # Create the submission file
    submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
    submission_df.to_csv(submission_filename, index=False)

    print(f"\nSubmission file '{submission_filename}' created successfully.")
    print("\nFirst 5 predictions in the submission file:")
    print(submission_df.head())

if __name__ == '__main__':
    run_training_and_prediction()
