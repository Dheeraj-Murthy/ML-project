import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# Import CatBoost Regressor
from catboost import CatBoostRegressor 

# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# --- Preprocessing and Feature Engineering ---

# Separate target variable and IDs
y = train_df['HotelValue']
train_ids = train_df['Id']
test_ids = test_df['Id']
train_df = train_df.drop(columns=['Id', 'HotelValue'])
test_df = test_df.drop(columns=['Id'])

# Combine for consistent processing
combined_df = pd.concat([train_df, test_df], axis=0, sort=False)

# Identify Categorical Columns
# CatBoost works best when you explicitly tell it which columns are categorical.
categorical_features_indices = combined_df.select_dtypes(include='object').columns.tolist()

# Handle Missing Values (REQUIRED by CatBoost for Categoricals)
# Fill missing categorical values with a special string
for col in categorical_features_indices:
    combined_df[col] = combined_df[col].fillna("MISSING")

# Fill missing numerical values with the median
numerical_cols = combined_df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    if combined_df[col].isnull().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())


# Separate back into training and testing sets
X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

# --- Train/Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Evaluation (CatBoost Regressor) ---

# Initialize the CatBoost model with optimal settings
cat_model = CatBoostRegressor(
    iterations=2500, # Max number of trees
    learning_rate=0.02, # Slightly aggressive learning rate
    depth=5, # Tree depth
    loss_function='RMSE',
    random_seed=42,
    # Pass the list of categorical feature names directly to the model!
    cat_features=categorical_features_indices, 
    verbose=0, # Suppress training output
    early_stopping_rounds=150, # Stop if validation loss doesn't improve
    thread_count=-1 # Use all available cores
)

# Train the CatBoost model
print("--- Training CatBoost Regressor (Best Possible Model) ---")
cat_model.fit(
    X_train, 
    y_train, 
    eval_set=(X_val, y_val),
)

# Make predictions on the validation set
val_predictions = cat_model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)

# Calculate scores
train_r2 = cat_model.score(X_train, y_train)
val_r2 = cat_model.score(X_val, y_val)

print("\n--- CatBoost Regressor Results ---")
print(f"Training R-squared: {train_r2}")
print(f"Validation R-squared: {val_r2}")
print(f"Validation Mean Squared Error: {val_mse}")


# --- Prediction and Submission ---

# Predict on the actual test data
test_predictions = cat_model.predict(X_test)

# Create the submission file
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
submission_df.to_csv('submission_catboost.csv', index=False)

print("Submission file 'submission_catboost.csv' created successfully.")
