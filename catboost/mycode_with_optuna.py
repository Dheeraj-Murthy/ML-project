import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from itertools import product

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
categorical_features_indices = combined_df.select_dtypes(
    include='object').columns.tolist()

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

# --- Define hyperparameter grid ---
learning_rates = [0.01, 0.02, 0.03]
depths = [4, 5, 6]
l2_leaf_regs = [1, 3, 5]
subsamples = [0.7, 0.8, 1.0]
rsm_values = [0.7, 0.8, 1.0]

# --- Loop over all combinations ---
best_rmse = float("inf")
best_params = None

for lr, depth, l2, subsample, rsm in product(
        learning_rates, depths, l2_leaf_regs, subsamples, rsm_values):
    print(
        f"\nTesting: lr={lr}, depth={depth}, l2={l2}, subsample={subsample}, rsm={rsm}")

    model = CatBoostRegressor(
        iterations=20000,
        learning_rate=lr,
        depth=depth,
        l2_leaf_reg=l2,
        subsample=subsample,
        rsm=rsm,
        border_count=254,
        bagging_temperature=1.0,
        random_strength=1.0,
        one_hot_max_size=2,
        cat_features=categorical_features_indices,
        thread_count=-1,
        loss_function='RMSE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=500,
        use_best_model=True
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds)
    r2 = model.score(X_val, y_val)

    print(f"Validation RMSE: {rmse:.4f}, R²: {r2:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = {
            'learning_rate': lr,
            'depth': depth,
            'l2_leaf_reg': l2,
            'subsample': subsample,
            'rsm': rsm
        }

print("\n=== Best Hyperparameters ===")
print(best_params)
print(f"Best Validation RMSE: {best_rmse:.4f}")
