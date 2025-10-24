import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Core Feature Engineering ---
def engineer_features(df):
    # Total square footage
    df['TotalSF'] = df['BasementTotalSF'] + df['GroundFloorArea'] + df['UpperFloorArea']
    
    # Room & Bath
    df['TotalBaths'] = df['FullBaths'] + 0.5 * df['HalfBaths'] + df['BasementFullBaths'] + 0.5 * df['BasementHalfBaths']
    df['AvgRoomSize'] = df['UsableArea'] / (df['TotalRooms'] + 1e-6)
    
    # Amenities (simplified)
    df['HasPool'] = (df['SwimmingPoolArea'] > 0).astype(int)
    df['HasTerrace'] = (df['TerraceArea'] > 0).astype(int)
    df['AmenitiesCount'] = df['HasPool'] + df['HasTerrace'] + (df['CentralAC'] == 'Y').astype(int)
    
    # Temporal
    df['PropertyAge'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRenovation'] = df['YearSold'] - df['RenovationYear']
    
    return df

# --- Load Data ---
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

y = train_df['HotelValue']
train_ids = train_df['Id']
test_ids = test_df['Id']

train_df = train_df.drop(columns=['Id', 'HotelValue'])
test_df = test_df.drop(columns=['Id'])

combined_df = pd.concat([train_df, test_df], axis=0, sort=False)

# --- Feature Engineering ---
combined_df = engineer_features(combined_df)

# --- Preprocessing ---
categorical_features = combined_df.select_dtypes(include='object').columns.tolist()
numerical_features = combined_df.select_dtypes(include=np.number).columns.tolist()

for col in categorical_features:
    combined_df[col] = combined_df[col].fillna("MISSING")
for col in numerical_features:
    combined_df[col] = combined_df[col].fillna(combined_df[col].median())

# Split back
X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- CatBoost Regressor ---
cat_model = CatBoostRegressor(
    iterations=2735,
    learning_rate=0.038,
    depth=5,
    l2_leaf_reg=4.011,
    random_strength=9.91,
    bagging_temperature=0.67,
    border_count=130,
    loss_function='RMSE',
    random_seed=42,
    cat_features=categorical_features,
    verbose=0,
    early_stopping_rounds=150,
    thread_count=-1
)

print("--- Training CatBoost Regressor (Reduced Features) ---")
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

# --- Evaluation ---
val_predictions = cat_model.predict(X_val)
train_r2 = cat_model.score(X_train, y_train)
val_r2 = r2_score(y_val, val_predictions)
val_mse = mean_squared_error(y_val, val_predictions)
val_rmse = np.sqrt(val_mse)

print("\n--- Model Evaluation ---")
print(f"Training R2: {train_r2}")
print(f"Validation R2: {val_r2}")
print(f"Validation MSE: {val_mse}")
print(f"Validation RMSE: {val_rmse}")

# --- Prediction and Submission ---
test_predictions = cat_model.predict(X_test)
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
submission_df.to_csv('submission_catboost_reduced.csv', index=False)
print("Submission CSV 'submission_catboost_reduced.csv' created successfully!")
