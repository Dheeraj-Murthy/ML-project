import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# --- Preprocessing and Feature Engineering ---

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

# Align columns after one-hot encoding, before scaling
train_cols = X.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X[c] = 0

X_test = X_test[train_cols]

# --- Feature Scaling ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# --- Model Training and Evaluation ---

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the RidgeCV Regression model
alphas_to_try = [0.1, 1.0, 10.0, 100.0]
model = RidgeCV(alphas=alphas_to_try)
model.fit(X_train, y_train)

# The best alpha found by RidgeCV
print(f"Best alpha found: {model.alpha_}")

# Make predictions on the validation set and calculate error
val_predictions = model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)
val_r2 = model.score(X_val, y_val)

# Calculate training scores
train_r2 = model.score(X_train, y_train)

print(f"Training R-squared: {train_r2}")
print(f"Validation R-squared: {val_r2}")
print(f"Validation Mean Squared Error: {val_mse}")


# --- Prediction and Submission ---

# Predict on the actual test data
test_predictions = model.predict(X_test_scaled)

# Create the submission file
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
submission_df.to_csv('submission_full_linear.csv', index=False)

print("Submission file 'submission_full_linear.csv' created successfully.")
