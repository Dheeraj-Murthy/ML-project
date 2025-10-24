import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Feature Engineering Function ---
def engineer_features(df):
    """Creates a comprehensive set of new features."""
    
    # Area Features
    df['TotalSF'] = df['BasementTotalSF'] + df['GroundFloorArea'] + df['UpperFloorArea']
    df['LandToBuildingRatio'] = df['LandArea'] / (df['GroundFloorArea'] + df['UpperFloorArea'] + 1e-6)
    df['BasementToTotalSF'] = df['BasementTotalSF'] / (df['TotalSF'] + 1e-6)
    df['PoolToLandRatio'] = df['SwimmingPoolArea'] / (df['LandArea'] + 1e-6)
    df['LowQualityAreaRatio'] = df['LowQualityArea'] / (df['UsableArea'] + 1e-6)
    df['BasementUnfinishedRatio'] = df['BasementUnfinishedSF'] / (df['BasementTotalSF'] + 1e-6)
    
    # Room & Bath Features
    df['TotalBaths'] = df['FullBaths'] + 0.5 * df['HalfBaths'] + df['BasementFullBaths'] + 0.5 * df['BasementHalfBaths']
    df['AvgRoomSize'] = df['UsableArea'] / (df['TotalRooms'] + 1e-6)
    df['GuestRoomRatio'] = df['GuestRooms'] / (df['TotalRooms'] + 1e-6)
    df['TotalRoomsIncludingKitchens'] = df['TotalRooms'] + df['Kitchens']
    df['TotalUsableAreaPerRoom'] = df['UsableArea'] / (df['TotalRoomsIncludingKitchens'] + 1e-6)
    df['BathroomsPerRoom'] = df['TotalBaths'] / (df['TotalRooms'] + 1e-6)
    
    # Parking / Driveway Features
    df['ParkingPerRoom'] = df['ParkingCapacity'] / (df['TotalRooms'] + 1e-6)
    df['ParkingAreaPerCar'] = df['ParkingArea'] / (df['ParkingCapacity'] + 1e-6)
    
    # Temporal Features
    df['PropertyAge'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRenovation'] = df['YearSold'] - df['RenovationYear']
    df['PropertyAgeSinceRenovation'] = np.maximum(0, df['YearsSinceRenovation'])
    df['SeasonSold'] = df['MonthSold'].apply(lambda x: (x-1)//3)
    
    # Interaction Features
    df['ExteriorRatio'] = df['FacadeArea'] / (df['GroundFloorArea'] + df['UpperFloorArea'] + 1e-6)
    df['RoadAccessPerLand'] = df['RoadAccessLength'] / (df['LandArea'] + 1e-6)
    
    # Binary / Presence Features
    df['HasBasement'] = (df['BasementTotalSF'] > 0).astype(int)
    df['HasPool'] = (df['SwimmingPoolArea'] > 0).astype(int)
    df['HasExtraFacility'] = (df['ExtraFacilityValue'] > 0).astype(int)
    df['HasSecondBasementFacility'] = df['BasementFacilityType2'].notna().astype(int)
    df['HasEnclosedVeranda'] = (df['EnclosedVerandaArea'] > 0).astype(int)
    df['HasTerrace'] = (df['TerraceArea'] > 0).astype(int)
    
    # Amenities Count
    df['AmenitiesCount'] = (
        df['HasPool'] +
        (df['BoundaryFence'] != 'None').astype(int) +
        df['HasTerrace'] +
        (df['ParkingArea'] > 0).astype(int) +
        (df['CentralAC'] == 'Y').astype(int)
    )
    
    # Combined Features
    df['TotalFacilities'] = df['BasementTotalSF'] + df['GuestRooms'] + df['Lounges'] + df['Kitchens']
    df['AreaPerFacility'] = df['UsableArea'] / (df['TotalFacilities'] + 1e-6)
    df['BasementFacilitiesArea'] = df['BasementFacilitySF1'] + df['BasementFacilitySF2']
    
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

# --- CatBoost Regressor with Best Hyperparameters ---
cat_model = CatBoostRegressor(
    iterations=2735,
    learning_rate=0.038,
    depth=5,
    l2_leaf_reg=4.011255855153269,
    random_strength=9.9076626,
    bagging_temperature=0.6710513693270637,
    border_count=130,
    loss_function='RMSE',
    random_seed=42,
    cat_features=categorical_features,
    verbose=0,
    early_stopping_rounds=150,
    thread_count=-1
)

print("--- Training CatBoost Regressor with Best Hyperparameters ---")
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
submission_df.to_csv('submission_catboost_bestparams.csv', index=False)
print("Submission CSV 'submission_catboost_bestparams.csv' created successfully!")
