import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error
import warnings
import os
import sys

# Suppress warnings for cleaner output during tuning
warnings.filterwarnings('ignore')

# Add the parent directory of PreProcessing to the path for module import
sys.path.append(os.path.join(os.path.dirname(__file__), '../PreProcessing'))
from preprocessing import *

# Import custom functions from the preprocessing module



def run_tuning_and_prediction():
    """
    Executes the XGBoost tuning, model fitting, and submission creation.
    """
    # Load and preprocess data (X_train_proc is a NumPy array)
    X_train_proc, X_test_proc, y_train_log, y_full_clean, test_ids, preprocessor = run_preprocessing(
        target_log_transform=True)

    # --- 2. Define Model and Hyperparameter Grid ---

    # Initialize the XGBoost Regressor
    xgb_model = XGBRegressor(random_state=42,
                             objective='reg:squarederror',
                             n_jobs=-1)

    # Define a smaller, discrete parameter grid for GridSearchCV (Exhaustive
    # Search)
    param_grid = {
        'n_estimators': [500, 1000],          # Reduced search space for speed
        'learning_rate': [0.03, 0.05, 0.1],   # Discrete values
        'max_depth': [4, 6],                  # Reduced search space
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'reg_lambda': [1.0, 2.0]
    }

    # --- 3. Run Grid Search Cross-Validation ---

    # We use KFold for robust cross-validation
    cv_folds = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search (Exhaustive search over the defined grid)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv_folds,
        verbose=1,
        n_jobs=-1
    )

    total_combinations = (len(param_grid['n_estimators']) *
                          len(param_grid['learning_rate']) *
                          len(param_grid['max_depth']) *
                          len(param_grid['subsample']) *
                          len(param_grid['colsample_bytree']) *
                          len(param_grid['reg_lambda']))

    print(
        f"\nStarting Grid Search for XGBoost Regressor ({total_combinations} total combinations, 5-fold CV)...")
    grid_search.fit(X_train_proc, y_train_log)
    print("Tuning complete.")

    # --- 4. Report and Save Results ---

    best_xgb_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    best_rmse_log = np.sqrt(best_score)

    # Create a Train-Validation Split for metric reporting (80/20 split on processed data)
    X_sub_train, X_val, y_sub_train_log, y_val_log = train_test_split(
        X_train_proc, y_train_log, test_size=0.2, random_state=42
    )
    # Need to split y_full_clean to get y_val_original
    _, y_val_original = train_test_split(
        y_full_clean, test_size=0.2, random_state=42
    )

    # Predict on X_val using the best model
    val_log_predictions = best_xgb_model.predict(X_val)
    val_predictions_original = np.expm1(val_log_predictions)
    rmse_original = np.sqrt(
        mean_squared_error(
            y_val_original,
            val_predictions_original))

    print("\n--- XGBoost Regressor Grid Search Results ---")
    print(f"Best CV Mean Squared Error (Log-transformed): {best_score:.5f}")
    print(
        f"Best CV Root Mean Squared Error (Log-transformed): {best_rmse_log:.5f}")
    print(f"Validation RMSE (Original Scale): {rmse_original:.4f}")
    save_model_results(
        os.path.basename(__file__),
        'XGBRegressor',
        rmse_original)
    print("\nBest Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Save the final best model (already fitted via CV)
    joblib.dump(best_xgb_model, 'fitted_xgboost_model_tuned.joblib')
    print("\nBest tuned XGBoost model saved to 'fitted_xgboost_model_tuned.joblib'")

    # --- 5. Final Metrics on Best Model (Optional for quick check) ---
    train_preds_log = best_xgb_model.predict(X_train_proc)
    train_mse = mean_squared_error(y_train_log, train_preds_log)

    print(f"\nFinal Training MSE (Log-transformed): {train_mse:.5f}")

    # --- 6. Prediction and Submission ---

    # Predict on the actual test data (log values)
    test_log_predictions = best_xgb_model.predict(X_test_proc)

    # Reverse the Log-transformation: HotelValue = exp(log_predictions) - 1
    test_predictions = np.expm1(test_log_predictions)

    # Create the submission file
    submission_filename = 'submission_xgboost_tuned.csv'
    submission_df = pd.DataFrame(
        {'Id': test_ids, 'HotelValue': test_predictions})
    submission_df.to_csv(submission_filename, index=False)

    print(
        f"\nSubmission file '{submission_filename}' created successfully in the current folder.")


if __name__ == '__main__':
    run_tuning_and_prediction()
