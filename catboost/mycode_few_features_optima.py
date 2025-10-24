import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import optuna

# --- Core Feature Engineering ---


def engineer_features(df):
    df['TotalSF'] = df['BasementTotalSF'] + \
        df['GroundFloorArea'] + df['UpperFloorArea']
    df['TotalBaths'] = df['FullBaths'] + 0.5 * df['HalfBaths'] + \
        df['BasementFullBaths'] + 0.5 * df['BasementHalfBaths']
    df['AvgRoomSize'] = df['UsableArea'] / (df['TotalRooms'] + 1e-6)
    df['HasPool'] = (df['SwimmingPoolArea'] > 0).astype(int)
    df['HasTerrace'] = (df['TerraceArea'] > 0).astype(int)
    df['AmenitiesCount'] = df['HasPool'] + \
        df['HasTerrace'] + (df['CentralAC'] == 'Y').astype(int)
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
combined_df = engineer_features(combined_df)

categorical_features = combined_df.select_dtypes(
    include='object').columns.tolist()
numerical_features = combined_df.select_dtypes(
    include=np.number).columns.tolist()

for col in categorical_features:
    combined_df[col] = combined_df[col].fillna("MISSING")
for col in numerical_features:
    combined_df[col] = combined_df[col].fillna(combined_df[col].median())

X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- Optuna Objective ---


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
        'loss_function': 'RMSE',
        'random_seed': 42
    }

    model = CatBoostRegressor(**params, cat_features=categorical_features)
    model.fit(
        X_train,
        y_train,
        eval_set=(
            X_val,
            y_val),
        early_stopping_rounds=150,
        verbose=False)

    preds = model.predict(X_val)
    return r2_score(y_val, preds)


# --- Run Optuna Study ---
print("--- Starting Hyperparameter Tuning (Reduced Features) ---")
study = optuna.create_study(direction='maximize')
# you can increase n_trials for better tuning
study.optimize(objective, n_trials=30)

print("Best Validation R2:", study.best_value)
print("Best Hyperparameters:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")

# --- Train Final Model with Best Params ---
best_params = study.best_params
best_params['cat_features'] = categorical_features
best_params['loss_function'] = 'RMSE'
best_params['random_seed'] = 42
best_params['verbose'] = 0

final_model = CatBoostRegressor(**best_params)
final_model.fit(X, y)

# --- Predict on Test ---
test_predictions = final_model.predict(X_test)
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
submission_df.to_csv('submission_catboost_reduced_optuna.csv', index=False)
print("Submission CSV 'submission_catboost_reduced_optuna.csv' created successfully!")
