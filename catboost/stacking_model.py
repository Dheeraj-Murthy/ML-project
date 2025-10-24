
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from PreProcessing.preprocessing import save_model_results

from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

# --- Data Loading and Preparation ---
print("--- Loading and Preparing Data ---")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

y = train_df['HotelValue']
test_ids = test_df['Id']

# Combine for consistent preprocessing
combined_df = pd.concat([train_df.drop(
    columns=['Id', 'HotelValue']), test_df.drop(columns=['Id'])], axis=0, sort=False)

# --- Preprocessing ---
categorical_features = combined_df.select_dtypes(
    include='object').columns.tolist()
numerical_features = combined_df.select_dtypes(
    include=np.number).columns.tolist()

for col in categorical_features:
    combined_df[col] = combined_df[col].fillna("MISSING")

for col in numerical_features:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)

# Create two versions of the data: one for CatBoost, one for XGB/LGBM
X_cat = combined_df.iloc[:len(train_df)]
X_test_cat = combined_df.iloc[len(train_df):]

X_xgb_lgb = pd.get_dummies(
    combined_df,
    columns=categorical_features,
    dummy_na=True)
X = X_xgb_lgb.iloc[:len(train_df)]
X_test = X_xgb_lgb.iloc[len(train_df):]
X_test = X_test[X.columns]  # Align columns

# --- Stacking Implementation ---
print("--- Starting Stacking Ensemble ---")

NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# --- Level 0 Models ---
# Out-of-fold predictions for training the meta-model
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))

# Test set predictions from each base model
test_preds_cat = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_lgb = np.zeros(len(X_test))

# --- CatBoost Model (from mycode.py) ---
cat_params = {
    'iterations': 2500,
    'learning_rate': 0.02,
    'depth': 5,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'cat_features': categorical_features,
    'verbose': 0,
    'early_stopping_rounds': 150
}

# --- XGBoost Model (tuned) ---
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 2423,
    'learning_rate': 0.04283,
    'max_depth': 3,
    'subsample': 0.7559,
    'colsample_bytree': 0.9819,
    'gamma': 4.0475,
    'lambda': 2.2385,
    'alpha': 0.6381,
    'seed': 42
}

# --- LightGBM Model (default params for diversity) ---
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'random_state': 42
}

# --- Training Level 0 Models with K-Fold ---
for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"===== Fold {fold + 1} =====")

    # --- CatBoost ---
    print("Training CatBoost...")
    X_train_cat, X_val_cat = X_cat.iloc[train_index], X_cat.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train_cat, y_train, eval_set=(X_val_cat, y_val))
    oof_cat[val_index] = cat_model.predict(X_val_cat)
    test_preds_cat += cat_model.predict(X_test_cat) / NFOLDS

    # --- XGBoost & LightGBM ---
    print("Training XGBoost and LightGBM...")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    oof_xgb[val_index] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test) / NFOLDS

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train)
    oof_lgb[val_index] = lgb_model.predict(X_val)
    test_preds_lgb += lgb_model.predict(X_test) / NFOLDS

# --- Level 1 Meta-Model ---
print("\n--- Training Meta-Model ---")

# Create new features from the predictions of the base models
X_meta = pd.DataFrame({
    'catboost': oof_cat,
    'xgboost': oof_xgb,
    'lightgbm': oof_lgb
})

X_test_meta = pd.DataFrame({
    'catboost': test_preds_cat,
    'xgboost': test_preds_xgb,
    'lightgbm': test_preds_lgb
})

meta_model = LinearRegression()
meta_model.fit(X_meta, y)

# --- Evaluation ---
meta_preds = meta_model.predict(X_meta)
final_r2 = r2_score(y, meta_preds)
print(f"\n--- Overall Stacking Results ---")
print(f"Final Stacking R-squared: {final_r2}")
final_rmse = np.sqrt(mean_squared_error(y, meta_preds))
save_model_results(
    os.path.basename(__file__),
    'Stacking Ensemble (CatBoost, XGBoost, LightGBM + LinearRegression)',
    final_rmse)

# --- Final Prediction ---
print("\n--- Generating Final Submission ---")
final_predictions = meta_model.predict(X_test_meta)

submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': final_predictions})
submission_df.to_csv('submission_stacking_model.csv', index=False)

print("Submission file 'submission_stacking_model.csv' created successfully!")
