
import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def engineer_features(df):
    """Creates a comprehensive set of new features."""
    # Area Features
    df['TotalSF'] = df['BasementTotalSF'] + df['GroundFloorArea'] + df['UpperFloorArea']

    # Room and Bath Features
    df['TotalBaths'] = df['FullBaths'] + 0.5 * df['HalfBaths'] + df['BasementFullBaths'] + 0.5 * df['BasementHalfBaths']
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

def objective(trial):
    """Objective function for Optuna hyperparameter tuning."""
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
    r2 = r2_score(y_val, preds)
    return r2

# --- Hyperparameter Tuning with Optuna ---
print("--- Starting Hyperparameter Tuning on Feature-Engineered Data ---")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print(f"  Value (Best R2): {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# --- Final Model Training ---
print("\n--- Training Final Model with Best Hyperparameters ---")
best_params = trial.params
best_params['cat_features'] = categorical_features
best_params['verbose'] = 0
best_params['random_seed'] = 42

final_model = CatBoostRegressor(**best_params)
final_model.fit(X, y)

# --- Prediction and Submission ---
print("\n--- Generating Final Submission ---")
final_predictions = final_model.predict(X_test)

submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': final_predictions})
submission_df.to_csv('submission_mycode_v3.csv', index=False)

print("Submission file 'submission_mycode_v3.csv' created successfully!")
