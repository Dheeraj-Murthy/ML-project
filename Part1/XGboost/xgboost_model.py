import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../PreProcessing'))

from preprocessing import save_model_results

# --- Load datasets ---
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# --- Preprocessing ---
y = train_df['HotelValue']
train_ids = train_df['Id']
test_ids = test_df['Id']

train_df = train_df.drop(columns=['Id', 'HotelValue'])
test_df = test_df.drop(columns=['Id'])

combined_df = pd.concat([train_df, test_df], axis=0, sort=False)

# Encode categorical columns
categorical_cols = combined_df.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    combined_df[col] = combined_df[col].fillna(
        "MISSING").astype('category').cat.codes

# Fill missing numerical values
numerical_cols = combined_df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    combined_df[col] = combined_df[col].fillna(combined_df[col].median())

# Split back into train/test
X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

# --- Train/Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- Initialize XGBoost Regressor ---
xgb_model = XGBRegressor(
    n_estimators=100000,
    learning_rate=0.022767,
    max_depth=5,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=3.0,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

# --- Train model with manual early stopping ---
best_score = float('inf')
best_iteration = 0
evals_result = {}

for i in range(1, xgb_model.get_params()['n_estimators'] + 1):
    xgb_model.set_params(n_estimators=i)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_preds = xgb_model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_preds)
    evals_result[i] = val_mse

    if val_mse < best_score:
        best_score = val_mse
        best_iteration = i
    else:
        break

# --- Final model evaluation ---
xgb_model.set_params(n_estimators=best_iteration)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_preds = xgb_model.predict(X_val)
val_mse = mean_squared_error(y_val, val_preds)
train_r2 = xgb_model.score(X_train, y_train)
val_r2 = xgb_model.score(X_val, y_val)

print("\n--- XGBoost Regressor Results ---")
print(f"Training R²: {train_r2:.4f}")
print(f"Validation R²: {val_r2:.4f}")
print(f"Validation RMSE: {np.sqrt(val_mse):.4f}")
save_model_results(
    os.path.basename(__file__),
    'XGBRegressor',
    np.sqrt(val_mse))

# --- Predict test set and create submission ---
test_preds = xgb_model.predict(X_test)
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_preds})
submission_df.to_csv('submission_xgboost.csv', index=False)
print("Submission file 'submission_xgboost.csv' created successfully.")
