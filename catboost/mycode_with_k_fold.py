import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

# --- Load Data ---
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Separate target and IDs
y = train_df['HotelValue']
train_ids = train_df['Id']
test_ids = test_df['Id']

train_df = train_df.drop(columns=['Id', 'HotelValue'])
test_df = test_df.drop(columns=['Id'])

# Combine for consistent preprocessing
combined_df = pd.concat([train_df, test_df], axis=0, sort=False)

# Identify categorical columns
categorical_features_indices = combined_df.select_dtypes(
    include='object').columns.tolist()

# Fill missing categorical and numerical values
for col in categorical_features_indices:
    combined_df[col] = combined_df[col].fillna("MISSING")

for col in combined_df.select_dtypes(include=np.number).columns:
    combined_df[col] = combined_df[col].fillna(combined_df[col].median())

# Split back into train and test
X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

# --- K-Fold Setup ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Define Parameters to Test ---
params = {
    "iterations": 20000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3,
    "subsample": 0.9,
    "rsm": 0.9,
    "loss_function": "RMSE",
    "cat_features": categorical_features_indices,
    "random_seed": 42,
    "early_stopping_rounds": 500,
    "verbose": 500,
    "thread_count": -1
}

# --- Cross-Validation Loop ---
rmse_scores = []
r2_scores = []

fold_num = 1
for train_idx, val_idx in kf.split(X, y):
    print(f"\n🧩 Fold {fold_num}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    val_preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)

    rmse_scores.append(rmse)
    r2_scores.append(r2)

    print(f"Fold {fold_num} RMSE: {rmse:.4f}")
    print(f"Fold {fold_num} R²: {r2:.4f}")
    fold_num += 1

# --- Average Performance ---
print("\n📊 --- Cross-Validation Summary ---")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

# --- Train Final Model on All Data ---
final_model = CatBoostRegressor(**params)
final_model.fit(X, y)

# --- Predict on Test Data ---
test_preds = final_model.predict(X_test)

# --- Save Submission ---
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_preds})
submission_df.to_csv('submission_catboost_kfold.csv', index=False)

print("\n✅ submission_catboost_kfold.csv created successfully.")
