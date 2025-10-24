import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold # CHANGED to GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint as sp_randint, uniform as sp_uniform # Kept for utility, but will be unused
import warnings
warnings.filterwarnings('ignore')

# Load the preprocessor and feature engineering function from the dedicated file
from preprocessing import run_preprocessing, major_feature_engineering, create_preprocessor_pipeline

# --- 1. Load and Prepare Data ---

# Load and preprocess data (X_train_proc is a NumPy array)
# We use log-transform (True) as it's best practice for price prediction with boosting models
X_train_proc, X_test_proc, y_train_log, test_ids, preprocessor = run_preprocessing(target_log_transform=True)

# --- 2. Define Model and Hyperparameter Grid ---

# Initialize the XGBoost Regressor
xgb_model = XGBRegressor(random_state=42, 
                         objective='reg:squarederror', 
                         n_jobs=-1)

# Define a smaller, discrete parameter grid for GridSearchCV (Exhaustive Search)
# Note: Grid Search should use fewer parameters and values than Randomized Search
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
grid_search = GridSearchCV( # CHANGED to GridSearchCV
    estimator=xgb_model, 
    param_grid=param_grid, # CHANGED to param_grid
    scoring='neg_mean_squared_error', 
    cv=cv_folds,
    verbose=1,
    n_jobs=-1
)

print(f"\nStarting Grid Search for XGBoost Regressor ({len(param_grid['n_estimators']) * len(param_grid['learning_rate']) * len(param_grid['max_depth']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['reg_lambda'])} total combinations, 5-fold CV)...")
grid_search.fit(X_train_proc, y_train_log)
print("Tuning complete.")

# --- 4. Report and Save Results ---

best_xgb_model = grid_search.best_estimator_ # CHANGED variable to grid_search
best_params = grid_search.best_params_
best_score = -grid_search.best_score_
best_rmse = np.sqrt(best_score)

print("\n--- XGBoost Regressor Grid Search Results ---") # CHANGED OUTPUT TEXT
print(f"Best CV Mean Squared Error (Log-transformed): {best_score:.5f}")
print(f"Best CV Root Mean Squared Error (Log-transformed): {best_rmse:.5f}")
print("\nBest Parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# Save the final best model (already fitted via CV)
joblib.dump(best_xgb_model, 'fitted_xgboost_model_tuned.joblib')
print("\nBest tuned XGBoost model saved to 'fitted_xgboost_model_tuned.joblib'")

# --- 5. Final Metrics on Best Model (Optional for quick check) ---
# We use the full training set metrics for an estimate of training performance
train_preds_log = best_xgb_model.predict(X_train_proc)
train_mse = mean_squared_error(y_train_log, train_preds_log)

print(f"\nFinal Training MSE (Log-transformed): {train_mse:.5f}")
