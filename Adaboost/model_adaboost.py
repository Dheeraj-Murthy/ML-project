import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor # Switched to AdaBoost Regressor
from sklearn.tree import DecisionTreeRegressor # Base estimator for AdaBoost
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import sys


from preprocessing import major_feature_engineering, create_and_fit_preprocessor, remove_outliers 

def run_model_training_and_prediction(train_file="train.csv", test_file="test.csv", target_log_transform=True):
    """
    Executes the full pipeline: data loading, preprocessing (using external joblib), 
    AdaBoost training, evaluation, and submission file creation.
    """
    try:
        # --- 1. Load Data ---
        # Assuming the train and test files are in the ../dataset/ folder relative to this script
        DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset') + os.sep
        train_df = pd.read_csv(DATA_PATH + train_file)
        test_df = pd.read_csv(DATA_PATH + test_file)
    except FileNotFoundError:
        print(f"ERROR: Could not find data files. Ensure they are in the '{DATA_PATH}' folder.")
        return

    test_ids = test_df['Id']
    
    # Separate features and target
    y_full = train_df['HotelValue']
    X_full = train_df.drop(columns=['Id', 'HotelValue'])
    X_test = test_df.drop(columns=['Id'])

    # 2. Outlier Removal (Only applied to training data)
    X_full_clean, y_full_clean = remove_outliers(X_full, y_full)

    # 3. Feature Engineering (Applied to clean train and raw test data)
    X_full_fe = major_feature_engineering(X_full_clean)
    X_test_fe = major_feature_engineering(X_test)

    # Ensure test columns align with train columns after FE
    X_test_fe = X_test_fe[X_full_fe.columns]

    # Apply target transformation
    y_full_train_log = np.log1p(y_full_clean) if target_log_transform else y_full_clean

    # Load the fitted preprocessor
    PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'PreProcessing', 'fitted_preprocessor.joblib')
    try:
        # --- FIX: Load joblib from the sibling preprocessing folder ---
        preprocessor = joblib.load(PREPROCESSOR_PATH) 
        print(f"Successfully loaded preprocessor from {PREPROCESSOR_PATH}.")
    except FileNotFoundError:
        print(f"ERROR: Could not find 'fitted_preprocessor.joblib' at {PREPROCESSOR_PATH}.")
        print("Please ensure preprocessing.py was run successfully and saved the file.")
        return

    # 4. Transform Data
    X_full_train_proc = preprocessor.transform(X_full_fe)
    X_test_proc = preprocessor.transform(X_test_fe)

    # Create a Train-Validation Split for metric reporting (80/20 split)
    X_sub_train, X_val, y_sub_train_log, y_val_log = train_test_split(
        X_full_train_proc, y_full_train_log, test_size=0.2, random_state=42
    )

    # --- 5. Define and Train the Model (AdaBoost Regressor) ---

    base_estimator = DecisionTreeRegressor(max_depth=4, random_state=42)

    ada_model = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=100, # Number of weak learners
        learning_rate=0.1, # Weight applied to each estimator at each boosting iteration
        random_state=42
    ) 

    print("\n--- Model Training & Evaluation (AdaBoost Regressor) ---")

    # Train the model on the sub-training split for evaluation
    ada_model.fit(X_sub_train, y_sub_train_log)

    # Calculate metrics (using log-transformed values)
    val_log_predictions = ada_model.predict(X_val)
    val_r2 = r2_score(y_val_log, val_log_predictions)
    val_mse = mean_squared_error(y_val_log, val_log_predictions)

    print(f"Validation R-squared (Log-transformed): {val_r2:.4f}")
    print(f"Validation Mean Squared Error (Log-transformed): {val_mse:.4f}")

    # --- 6. Create Final Production Model and Save ---

    print("\nFitting final production model on ALL training data...")

    # Re-train the AdaBoost model on the FULL processed data
    ada_model.fit(X_full_train_proc, y_full_train_log)

    # Save the trained model object (saved in the Adaboost folder)
    joblib.dump(ada_model, 'fitted_adaboost_model.joblib')

    print("Final trained AdaBoost model saved to 'fitted_adaboost_model.joblib'.")

    # --- 7. Prediction and Submission ---

    # Predict on the actual test data (log values)
    test_log_predictions = ada_model.predict(X_test_proc)

    # Reverse the Log-transformation: HotelValue = exp(log_predictions) - 1
    test_predictions = np.expm1(test_log_predictions)

    # Create the submission file (saved in the Adaboost folder)
    submission_filename = 'submission_adaboost.csv'
    submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
    submission_df.to_csv(submission_filename, index=False)

    print(f"\nSubmission file '{submission_filename}' created successfully in the adaboost folder.")


if __name__ == '__main__':
    # Execute the full pipeline
    run_model_training_and_prediction(target_log_transform=True)
