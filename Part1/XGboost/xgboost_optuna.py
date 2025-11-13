import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import optuna
import os
from PreProcessing.preprocessing import save_model_results

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

# --- Objective function for Optuna ---


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 60000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        "objective": "reg:squarederror"
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds)  # Use RMSE directly
    return rmse


# --- Run Optuna study ---
study = optuna.create_study(direction="minimize")
# Adjust n_trials as needed
study.optimize(objective, n_trials=500, show_progress_bar=True)

# --- Best trial ---
trial = study.best_trial
print("Best trial:")
print(f"  RMSE: {trial.value}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# --- Train final model with best hyperparameters ---
best_params = trial.params
best_model = XGBRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1,
    objective="reg:squarederror")
best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_preds = best_model.predict(X_val)
val_rmse = mean_squared_error(y_val, val_preds)
train_r2 = best_model.score(X_train, y_train)
val_r2 = best_model.score(X_val, y_val)

print("\n--- Final Model Results ---")
print(f"Training R²: {train_r2:.4f}")
print(f"Validation R²: {val_r2:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
save_model_results(
    os.path.basename(__file__),
    'XGBRegressor (Optuna)',
    val_rmse)

# --- Predict test set and create submission ---
test_preds = best_model.predict(X_test)
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_preds})
submission_df.to_csv('submission_xgboost_optuna.csv', index=False)
print("Submission file 'submission_xgboost_optuna.csv' created successfully.")
