
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def engineer_features(df):
    """Creates a comprehensive set of new features."""
    # Area Features
    df['TotalSF'] = df['BasementTotalSF'] + df['GroundFloorArea'] + df['UpperFloorArea']

    # Room and Bath Features
    df['TotalBaths'] = df['FullBaths'] + 0.5 * df['HalfBaths'] + df['BasementFullBaths'] + 0.5 * df['BasementHalfBaths']
    # Handle division by zero for rooms
    df['AvgRoomSize'] = df['UsableArea'] / (df['TotalRooms'] + 1e-6)
    df['GuestRoomRatio'] = df['GuestRooms'] / (df['TotalRooms'] + 1e-6)

    # Amenities Count
    df['AmenitiesCount'] = (
        (df['SwimmingPoolArea'] > 0).astype(int) + 
        (df['BoundaryFence'] != 'None').astype(int) + 
        (df['TerraceArea'] > 0).astype(int) + 
        (df['ParkingArea'] > 0).astype(int) + 
        (df['CentralAC'] == 'Y').astype(int)
    )
    return df

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Separate target variable and IDs
y = train_df['HotelValue']
test_ids = test_df['Id']

# Combine for consistent processing
combined_df = pd.concat([train_df.drop(columns=['Id', 'HotelValue']), test_df.drop(columns=['Id'])], axis=0, sort=False)

# --- Feature Engineering ---
print("--- Engineering New Features ---")
combined_df = engineer_features(combined_df)

# --- Preprocessing ---
categorical_features = combined_df.select_dtypes(include='object').columns.tolist()
numerical_features = combined_df.select_dtypes(include=np.number).columns.tolist()

for col in categorical_features:
    combined_df[col] = combined_df[col].fillna("MISSING")

for col in numerical_features:
    if combined_df[col].isnull().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

# Separate back into training and testing sets
X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

# --- Train/Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Evaluation ---
print("--- Training CatBoost with All New Features ---")
cat_model = CatBoostRegressor(
    iterations=2500,
    learning_rate=0.02,
    depth=5,
    loss_function='RMSE',
    random_seed=42,
    cat_features=categorical_features,
    verbose=0,
    early_stopping_rounds=150
)

cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

# --- Evaluation ---
val_predictions = cat_model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)
train_r2 = cat_model.score(X_train, y_train)
val_r2 = r2_score(y_val, val_predictions)

print("\n--- Results with All Features ---")
print(f"Training R-squared: {train_r2}")
print(f"Validation R-squared: {val_r2}")
print(f"Validation Mean Squared Error: {val_mse}")

# --- Prediction and Submission ---
print("\n--- Generating Final Submission ---")
test_predictions = cat_model.predict(X_test)

submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
submission_df.to_csv('submission_mycode_v2.csv', index=False)

print("Submission file 'submission_mycode_v2.csv' created successfully.")
