import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import optuna

# --- Load Data ---
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# --- Preprocessing ---
y = train_df['HotelValue']
train_ids = train_df['Id']
test_ids = test_df['Id']
train_df = train_df.drop(columns=['Id', 'HotelValue'])
test_df = test_df.drop(columns=['Id'])

combined_df = pd.concat([train_df, test_df], axis=0, sort=False)

categorical_features_indices = combined_df.select_dtypes(include='object').columns.tolist()

# Fill missing categorical values
for col in categorical_features_indices:
    combined_df[col] = combined_df[col].fillna("MISSING")

# Fill missing numerical values
for col in combined_df.select_dtypes(include=np.number).columns:
    if combined_df[col].isnull().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

# --- Define Objective Function for Optuna ---
def objective(trial):
    params = {
        "iterations": 10000,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0.1, 2.0),
        "loss_function": "RMSE",
        "cat_features": categorical_features_indices,
        "verbose": 0,
        "random_seed": 42
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200, verbose=0)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# --- Run Optuna Study ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\n✅ Best Parameters Found:")
print(study.best_params)
print(f"Best Average RMSE: {study.best_value}")

# --- Train Final Model on All Data using Best Params ---
best_params = study.best_params
best_params.update({
    "iterations": 10000,
    "loss_function": "RMSE",
    "cat_features": categorical_features_indices,
    "verbose": 500
})

final_model = CatBoostRegressor(**best_params)
final_model.fit(X, y)

# --- Make Final Predictions ---
test_predictions = final_model.predict(X_test)
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
submission_df.to_csv('submission_optuna_catboost.csv', index=False)

print("\n🎯 Final model trained with best hyperparameters!")
print("Submission file 'submission_optuna_catboost.csv' created successfully.")
