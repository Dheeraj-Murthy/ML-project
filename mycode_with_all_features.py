import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------
# Feature Engineering Function
# --------------------------
def engineer_features(df):
    """Comprehensive features including golden features for property value."""
    
    # ----- Core Area & Room Features -----
    df['TotalSF'] = df['BasementTotalSF'] + df['GroundFloorArea'] + df['UpperFloorArea']
    df['LandToBuildingRatio'] = df['LandArea'] / (df['GroundFloorArea'] + df['UpperFloorArea'] + 1e-6)
    df['BasementToTotalSF'] = df['BasementTotalSF'] / (df['TotalSF'] + 1e-6)
    df['TotalSFPerRoom'] = df['TotalSF'] / (df['TotalRooms'] + 1e-6)
    df['AvgRoomSize'] = df['UsableArea'] / (df['TotalRooms'] + 1e-6)
    df['LowQualityAreaRatio'] = df['LowQualityArea'] / (df['UsableArea'] + 1e-6)
    
    # ----- Bathroom & Kitchen Features -----
    df['TotalBaths'] = df['FullBaths'] + 0.5*df['HalfBaths'] + df['BasementFullBaths'] + 0.5*df['BasementHalfBaths']
    df['BathroomsPerRoom'] = df['TotalBaths'] / (df['TotalRooms'] + 1e-6)
    df['GuestRoomRatio'] = df['GuestRooms'] / (df['TotalRooms'] + 1e-6)
    df['TotalRoomsIncludingKitchens'] = df['TotalRooms'] + df['Kitchens']
    df['TotalUsableAreaPerRoom'] = df['UsableArea'] / (df['TotalRoomsIncludingKitchens'] + 1e-6)
    df['KitchensPerRoom'] = df['Kitchens'] / (df['TotalRooms'] + 1e-6)
    df['BathroomsPerKitchens'] = df['TotalBaths'] / (df['Kitchens'] + 1e-6)
    
    # ----- Parking / Driveway Features -----
    df['ParkingPerRoom'] = df['ParkingCapacity'] / (df['TotalRooms'] + 1e-6)
    df['ParkingAreaPerCar'] = df['ParkingArea'] / (df['ParkingCapacity'] + 1e-6)
    
    # ----- Temporal Features -----
    df['PropertyAge'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRenovation'] = df['YearSold'] - df['RenovationYear']
    df['PropertyAgeSinceRenovation'] = np.maximum(0, df['YearsSinceRenovation'])
    df['AgePerRoom'] = df['PropertyAge'] / (df['TotalRooms'] + 1e-6)
    df['RenovationRatio'] = df['YearsSinceRenovation'] / (df['PropertyAge'] + 1e-6)
    df['SeasonSold'] = df['MonthSold'].apply(lambda x: (x-1)//3)
    
    # ----- Interaction Features -----
    df['ExteriorRatio'] = df['FacadeArea'] / (df['GroundFloorArea'] + df['UpperFloorArea'] + 1e-6)
    df['RoadAccessPerLand'] = df['RoadAccessLength'] / (df['LandArea'] + 1e-6)
    df['RoadAccessPerRoom'] = df['RoadAccessLength'] / (df['TotalRooms'] + 1e-6)
    df['FacadePerTotalSF'] = df['FacadeArea'] / (df['TotalSF'] + 1e-6)
    df['ParkingToFacade'] = df['ParkingArea'] / (df['FacadeArea'] + 1e-6)
    df['PoolToLandRatio'] = df['SwimmingPoolArea'] / (df['LandArea'] + 1e-6)
    
    # ----- Binary / Presence Features -----
    df['HasBasement'] = (df['BasementTotalSF'] > 0).astype(int)
    df['HasPool'] = (df['SwimmingPoolArea'] > 0).astype(int)
    df['HasExtraFacility'] = (df['ExtraFacilityValue'] > 0).astype(int)
    df['HasSecondBasementFacility'] = df['BasementFacilityType2'].notna().astype(int)
    df['HasEnclosedVeranda'] = (df['EnclosedVerandaArea'] > 0).astype(int)
    df['HasTerrace'] = (df['TerraceArea'] > 0).astype(int)
    df['HasAnyFacility'] = ((df['HasPool'] | df['HasTerrace'] | df['HasBasement'] | df['HasExtraFacility']).astype(int))
    
    # ----- Amenities Count Features -----
    df['AmenitiesCount'] = df['HasPool'] + df['HasTerrace'] + (df['ParkingArea']>0).astype(int) + (df['CentralAC']=='Y').astype(int) + (df['BoundaryFence']!='None').astype(int)
    df['LuxuryFeaturesCount'] = df['AmenitiesCount'] + df['GuestRooms'] + df['Lounges']
    
    # ----- Combined / Golden Features -----
    df['TotalFacilities'] = df['BasementTotalSF'] + df['GuestRooms'] + df['Lounges'] + df['Kitchens']
    df['AreaPerFacility'] = df['UsableArea'] / (df['TotalFacilities'] + 1e-6)
    df['BasementFacilitiesArea'] = df['BasementFacilitySF1'] + df['BasementFacilitySF2']
    
    return df

# --------------------------
# Load Data
# --------------------------
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

y = train_df['HotelValue']
train_ids = train_df['Id']
test_ids = test_df['Id']

train_df = train_df.drop(columns=['Id', 'HotelValue'])
test_df = test_df.drop(columns=['Id'])

combined_df = pd.concat([train_df, test_df], axis=0, sort=False)

# --------------------------
# Feature Engineering
# --------------------------
combined_df = engineer_features(combined_df)

# --------------------------
# Preprocessing
# --------------------------
categorical_features = combined_df.select_dtypes(include='object').columns.tolist()
numerical_features = combined_df.select_dtypes(include=np.number).columns.tolist()

for col in categorical_features:
    combined_df[col] = combined_df[col].fillna("MISSING")
for col in numerical_features:
    combined_df[col] = combined_df[col].fillna(combined_df[col].median())

# Split back into train/test
X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Optuna Hyperparameter Tuning
# --------------------------
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 4000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'depth': trial.suggest_int('depth', 4, 7),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 128, 255),
        'verbose': 0,
        'eval_metric': 'R2',
        'random_seed': 42
    }
    model = CatBoostRegressor(**params, cat_features=categorical_features)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=150, verbose=False)
    preds = model.predict(X_val)
    return r2_score(y_val, preds)

print("--- Starting Optuna Hyperparameter Tuning ---")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

trial = study.best_trial
print(f"\nBest R2 on Validation: {trial.value}")
print("Best Hyperparameters:")
for key, value in trial.params.items():
    print(f"  {key}: {value}")

# --------------------------
# Train Final Model
# --------------------------
best_params = trial.params
best_params['cat_features'] = categorical_features
best_params['verbose'] = 0
best_params['random_seed'] = 42

final_model = CatBoostRegressor(**best_params)
final_model.fit(X, y)

# --------------------------
# Evaluation
# --------------------------
train_r2 = final_model.score(X, y)
val_predictions = final_model.predict(X_val)
val_r2 = r2_score(y_val, val_predictions)
val_mse = mean_squared_error(y_val, val_predictions)
val_rmse = np.sqrt(val_mse)

print("\n--- Final Model Evaluation ---")
print(f"Training R2: {train_r2}")
print(f"Validation R2: {val_r2}")
print(f"Validation MSE: {val_mse}")
print(f"Validation RMSE: {val_rmse}")

# --------------------------
# Prediction & Submission
# --------------------------
test_predictions = final_model.predict(X_test)
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
submission_df.to_csv('submission_feature_engineered_catboost_full.csv', index=False)
print("\nSubmission CSV 'submission_feature_engineered_catboost_full.csv' created successfully!")
