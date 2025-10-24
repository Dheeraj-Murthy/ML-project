import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
# Import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score # ADDED r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# Define a function to encapsulate the training and prediction pipeline
def run_tuning_and_prediction():
    # --- 1. Load the datasets ---
    # Define paths relative to the current script's location
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset') + os.sep

    try:
        # Load from the expected project structure path
        train_df = pd.read_csv(DATA_PATH + 'train.csv')
        test_df = pd.read_csv(DATA_PATH + 'test.csv')
    except FileNotFoundError:
        print(f"ERROR: Ensure train.csv and test.csv are in the {DATA_PATH} directory.")
        return

    # --- 2. Preprocessing and Feature Engineering (Self-Contained) ---

    # Separate target variable and IDs
    y = train_df['HotelValue']
    test_ids = test_df['Id']
    train_df = train_df.drop(columns=['Id', 'HotelValue'])
    X_test_raw = test_df.drop(columns=['Id']) # Keep test raw data for preprocessing

    # Log-transform the target variable (CRITICAL for skewed price data)
    y_log = np.log1p(y)

    # Combine for consistent processing
    combined_df = pd.concat([train_df, X_test_raw], axis=0, sort=False)

    print("\nStarting self-contained preprocessing steps...")

    # Handle Missing Values
    # Numerical columns: fill with the median
    for col in combined_df.select_dtypes(include=np.number).columns:
        if combined_df[col].isnull().any():
            combined_df[col] = combined_df[col].fillna(combined_df[col].median())

    # Categorical columns: fill with the mode
    for col in combined_df.select_dtypes(include='object').columns:
        if combined_df[col].isnull().any():
            # Use mode[0] as mode can return multiple values
            combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

    # One-Hot Encode Categorical Features
    combined_df = pd.get_dummies(combined_df, drop_first=True)

    # Separate back into training and testing sets
    X = combined_df.iloc[:len(train_df)]
    X_test = combined_df.iloc[len(train_df):]

    # Align columns after one-hot encoding (ensure same features in both sets)
    train_cols = X.columns
    test_cols = X_test.columns

    # Add missing columns (all zeros) to the test set
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0

    # Add missing columns (all zeros) to the training set (less common, but safe)
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X[c] = 0

    # Ensure test columns order matches train columns order
    X_test = X_test[train_cols]
    
    # --- Feature Scaling (Used for consistency, though less critical for tree models) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data ready. Training data shape: {X_scaled.shape}, Test data shape: {X_test_scaled.shape}")


    # --- 3. Model Training and Evaluation (Random Forest Hyperparameter Tuning) ---

    # Split the training data for cross-validation and final validation check
    X_train, X_val, y_train_log, y_val_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

    # Define the fixed parameter grid to search over for Grid Search
    param_grid = {
        'n_estimators': [100, 200], # 2 values
        'max_depth': [10, 15, 20],   # 3 values
        'min_samples_split': [10]    # 1 value
    } # Total 6 combinations

    # Initialize the base Random Forest model
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Initialize GridSearchCV (6 combinations * 3 CV folds = 18 models trained)
    grid_search = GridSearchCV(
        estimator=rf_model, 
        param_grid=param_grid, 
        # MSE on log-transformed target is the standard metric
        scoring='neg_mean_squared_error', 
        cv=3, 
        verbose=1, 
        n_jobs=-1
    )

    print("\nStarting Grid Search for best Random Forest parameters (Testing 6 combinations)...")
    grid_search.fit(X_train, y_train_log)

    # The best model found during the search
    best_model = grid_search.best_estimator_

    print("\n--- Optimized Random Forest Regressor Results (Grid Search Minimal) ---")
    print(f"Best Parameters found: {grid_search.best_params_}")

    # Calculate metrics on the hold-out validation set (X_val)
    val_log_predictions = best_model.predict(X_val)
    
    # Calculate R2 and MSE using log-transformed values
    val_r2 = r2_score(y_val_log, val_log_predictions)
    val_mse = mean_squared_error(y_val_log, val_log_predictions)

    print(f"Validation R-squared (Log-transformed, Best Model): {val_r2:.4f}")
    print(f"Validation Mean Squared Error (Log-transformed, Best Model): {val_mse:.4f}")
    
    # --- 4. Prediction and Submission ---

    # Predict on the actual test data using the best model (log values)
    test_log_predictions = best_model.predict(X_test_scaled)

    # Reverse the Log-transformation: HotelValue = exp(log_predictions) - 1
    test_predictions = np.expm1(test_log_predictions)

    # Save the best model
    model_filename = 'fitted_random_forest_tuned.joblib'
    joblib.dump(best_model, model_filename)
    print(f"\nBest tuned Random Forest model saved to '{model_filename}'.")


    # Create the submission file
    submission_filename = 'submission_random_forest_grid_search_minimal.csv'
    submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
    submission_df.to_csv(submission_filename, index=False)

    print(f"\nSubmission file '{submission_filename}' created successfully.")
    print("\nFirst 5 predictions in the submission file:")
    print(submission_df.head())


if __name__ == '__main__':
    run_tuning_and_prediction()
