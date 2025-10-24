import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from catboost import CatBoostRegressor

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
categorical_features_indices = combined_df.select_dtypes(include='object').columns.tolist()

for col in categorical_features_indices:
    combined_df[col] = combined_df[col].fillna("MISSING")

numerical_cols = combined_df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    combined_df[col] = combined_df[col].fillna(combined_df[col].median())

X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

# --- Train/Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Improved CatBoost model ---
cat_model = CatBoostRegressor(
    iterations=20000,              # Increased for lower learning rate
    learning_rate=0.02,            # Slightly lower for stable training
    depth=5,
    loss_function='RMSE',
    random_seed=42,
    
    # Regularization & randomness
    l2_leaf_reg=3.0,
    subsample=0.8,                 # slightly less than 1 for bagging
    rsm=0.8,                       # feature subsample
    bagging_temperature=1.0,
    random_strength=1.0,
    border_count=255,
    one_hot_max_size=2,
    
    cat_features=categorical_features_indices,
    
    # Computation
    thread_count=-1,
    verbose=100,                   # prints training progress every 100 iterations
    
    # Early stopping to avoid overfitting
    early_stopping_rounds=500,
    use_best_model=True
)

# --- Train ---
print("--- Training CatBoost Regressor ---")
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

# --- Validation Performance ---
val_preds = cat_model.predict(X_val)
val_mse = mean_squared_error(y_val, val_preds)
train_r2 = cat_model.score(X_train, y_train)
val_r2 = cat_model.score(X_val, y_val)

print("\n--- CatBoost Results ---")
print(f"Training R²: {train_r2:.4f}")
print(f"Validation R²: {val_r2:.4f}")
print(f"Validation RMSE: {np.sqrt(val_mse):.4f}")

# --- Predict test set and create submission ---
test_preds = cat_model.predict(X_test)
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_preds})
submission_df.to_csv('submission_catboost.csv', index=False)
print("Submission file 'submission_catboost.csv' created successfully.")
