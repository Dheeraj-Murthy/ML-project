import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission_df = pd.read_csv('sample_submission.csv')
except FileNotFoundError:
    print("Ensure train.csv, test.csv, and sample_submission.csv are in the same directory.")
    exit()

# A simple approach to handle missing values and select features
def preprocess(df):
    # Select only numeric columns for simplicity
    numeric_df = df.select_dtypes(include=[np.number])
    # Fill missing values with the mean of the column
    for col in numeric_df.columns:
        if numeric_df[col].isnull().any():
            numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())
    return numeric_df

train_processed = preprocess(train_df)
test_processed = preprocess(test_df)

# Define features (X) and target (y)
features = [col for col in train_processed.columns if col not in ['Id', 'HotelValue']]
X = train_processed[features]
y = train_processed['HotelValue']

# Align columns - crucial for consistent feature sets
train_cols = X.columns
test_cols = [col for col in test_processed.columns if col in features]
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    test_processed[c] = 0 # Or some other default value
test_processed = test_processed[train_cols] # Ensure order is the same

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set and calculate error
val_predictions = model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)
val_r2 = model.score(X_val, y_val)

# Calculate training scores
train_r2 = model.score(X_train, y_train)

print(f"Training R-squared: {train_r2}")
print(f"Validation R-squared: {val_r2}")
print(f"Validation Mean Squared Error: {val_mse}")

# Predict on the actual test data
test_predictions = model.predict(test_processed)

# Create the submission file
submission_df = pd.DataFrame({'Id': test_df['Id'], 'HotelValue': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
