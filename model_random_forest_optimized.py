import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
# Import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the datasets
try:
    # Assuming the data files are now inside the ML-project directory or accessible
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Ensure train.csv and test.csv are accessible from the current directory.")
    exit()

# --- Preprocessing and Feature Engineering (Identical to Linear Models) ---

# Separate target variable
y = train_df['HotelValue']
train_ids = train_df['Id']
test_ids = test_df['Id']
train_df = train_df.drop(columns=['Id', 'HotelValue'])
test_df = test_df.drop(columns=['Id'])

# Combine for consistent processing
combined_df = pd.concat([train_df, test_df], axis=0, sort=False)

# Handle Missing Values
# Numerical columns: fill with the median
for col in combined_df.select_dtypes(include=np.number).columns:
    if combined_df[col].isnull().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

# Categorical columns: fill with the mode
for col in combined_df.select_dtypes(include='object').columns:
    if combined_df[col].isnull().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

# One-Hot Encode Categorical Features
combined_df = pd.get_dummies(combined_df, drop_first=True)

# Separate back into training and testing sets
X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

# Align columns after one-hot encoding
train_cols = X.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X[c] = 0

X_test = X_test[train_cols]

# --- Feature Scaling (Used for consistency, though not required for RF) ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# --- Model Training and Evaluation (Random Forest Hyperparameter Tuning) ---

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the fixed parameter grid to search over for Grid Search (6 combinations)
param_grid = {
    'n_estimators': [100, 200], # 2 values
    'max_depth': [10, 15, 20],   # 3 values
    'min_samples_split': [10]    # 1 value
}

# Initialize the base Random Forest model
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

# Initialize GridSearchCV (6 combinations * 3 CV folds = 18 models trained)
grid_search = GridSearchCV(
    estimator=rf_model, 
    param_grid=param_grid, 
    scoring='neg_mean_squared_error', 
    cv=3, 
    verbose=2, 
    n_jobs=-1
)

print("Starting Grid Search for best Random Forest parameters (Testing 6 combinations)...")
grid_search.fit(X_train, y_train)

# The best model found during the search
best_model = grid_search.best_estimator_

print("\n--- Optimized Random Forest Regressor Results (Grid Search Minimal) ---")
print(f"Best Parameters found: {grid_search.best_params_}")

# Make predictions on the external validation set (X_val)
val_predictions = best_model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)

# Calculate scores
train_r2 = best_model.score(X_train, y_train)
val_r2 = best_model.score(X_val, y_val)

print(f"Training R-squared (Best Model): {train_r2}")
print(f"Validation R-squared (Best Model): {val_r2}")
print(f"Validation Mean Squared Error (Best Model): {val_mse}")


# --- Prediction and Submission ---

# Predict on the actual test data using the best model
test_predictions = best_model.predict(X_test_scaled)

# Create the submission file
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
# Saving with a unique name
submission_df.to_csv('submission_random_forest_grid_search_minimal.csv', index=False)

print("\nSubmission file 'submission_random_forest_grid_search_minimal.csv' created successfully.")