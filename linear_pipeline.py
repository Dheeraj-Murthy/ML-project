import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def major_feature_engineering(df):
    """
    Performs feature engineering for the Hotel Value dataset.
    Creates new features and combines related features.
    """
    df = df.copy()

    # Total Usable Area
    df['TotalSF'] = (df['GroundFloorArea'] + 
                     df['UpperFloorArea'] + 
                     df['BasementTotalSF'])

    # Temporal Features
    df['Age'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRemodel'] = df['YearSold'] - df['RenovationYear']
    df['YearsSinceRemodel'] = np.where(df['YearsSinceRemodel'] < 0, 0, df['YearsSinceRemodel'])
    df.loc[df['RenovationYear'] == 0, 'YearsSinceRemodel'] = df['Age']

    # Quality and Condition Scores
    df['OverallScore'] = (df['OverallQuality'] + df['OverallCondition']) / 2.0

    # Total Bathrooms
    df['TotalBaths'] = (df['FullBaths'] + 0.5 * df['HalfBaths'] +
                        df['BasementFullBaths'] + 0.5 * df['BasementHalfBaths'])

    # Interaction Features
    df['Qual_x_GroundSF'] = df['OverallQuality'] * df['GroundFloorArea']

    # Drop original columns that have been combined
    drop_cols = ['GroundFloorArea', 'UpperFloorArea', 'BasementTotalSF',
                 'ConstructionYear', 'RenovationYear', 'OverallQuality',
                 'OverallCondition', 'FullBaths', 'HalfBaths',
                 'BasementFullBaths', 'BasementHalfBaths']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    return df


def create_preprocessor_pipeline(X_train):
    """
    Creates a ColumnTransformer preprocessing pipeline.
    Numerical: median imputation + scaling
    Categorical: 'None' imputation + one-hot encoding
    """
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def run_preprocessing(train_file="train.csv", test_file="test.csv", target_log_transform=True):
    """
    Main preprocessing function: loads data, engineers features, and transforms.
    Returns processed train/test data and fitted preprocessor.
    """
    # Load datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    test_ids = test_df['Id']

    # Separate features and target
    y_full_train = train_df['HotelValue']
    X_full_train = train_df.drop(['Id', 'HotelValue'], axis=1)
    X_test = test_df.drop('Id', axis=1)

    # Apply target transformation
    y_train_transformed = np.log1p(y_full_train) if target_log_transform else y_full_train

    # Apply feature engineering
    X_full_train_fe = major_feature_engineering(X_full_train)
    X_test_fe = major_feature_engineering(X_test)
    X_test_fe = X_test_fe[X_full_train_fe.columns]

    # Create and fit preprocessor
    preprocessor = create_preprocessor_pipeline(X_full_train_fe)
    preprocessor.fit(X_full_train_fe)
    joblib.dump(preprocessor, 'fitted_preprocessor.joblib')

    # Transform data
    X_train_processed = preprocessor.transform(X_full_train_fe)
    X_test_processed = preprocessor.transform(X_test_fe)

    print("Data Preprocessing Complete")
    print(f"Preprocessor saved to 'fitted_preprocessor.joblib'")
    print(f"Processed training shape: {X_train_processed.shape}")

    return X_train_processed, X_test_processed, y_train_transformed, test_ids, preprocessor


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_and_evaluate_model(train_file="train.csv", test_file="test.csv"):
    """
    Trains a Linear Regression model with validation metrics.
    Saves the trained model for prediction.
    """
    print("\n=== Starting Model Training ===\n")
    
    # Load and preprocess data
    X_full_train, X_test, y_full_train_log, test_ids, preprocessor = run_preprocessing(
        train_file=train_file,
        test_file=test_file,
        target_log_transform=True
    )

    # Create train-validation split
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X_full_train, y_full_train_log,
        test_size=0.2,
        random_state=42
    )

    # Train model on training subset
    print("\n--- Training Linear Regression Model ---")
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train_log)

    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    val_r2 = r2_score(y_val_log, val_predictions)
    val_mse = mean_squared_error(y_val_log, val_predictions)

    print(f"\nValidation Metrics (Log-transformed):")
    print(f"  R² Score: {val_r2:.4f}")
    print(f"  MSE:      {val_mse:.4f}")

    # Retrain on full dataset
    print("\n--- Training Final Model on Full Dataset ---")
    model.fit(X_full_train, y_full_train_log)

    # Save model
    model_filename = 'fitted_linear_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\nModel saved to '{model_filename}'")
    
    return model, preprocessor


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def run_prediction(test_file="test.csv",
                  preprocessor_file="fitted_preprocessor.joblib",
                  model_file="fitted_linear_model.joblib",
                  submission_filename="submission_linear.csv"):
    """
    Generates predictions and creates submission file.
    """
    print("\n=== Starting Prediction Process ===\n")
    
    # Load test data
    try:
        test_df = pd.read_csv(test_file)
        test_ids = test_df['Id']
        X_test = test_df.drop('Id', axis=1)
        print(f"Loaded test data: {len(test_df)} rows")
    except FileNotFoundError:
        print(f"Error: '{test_file}' not found")
        return

    # Load preprocessor and model
    try:
        preprocessor = joblib.load(preprocessor_file)
        model = joblib.load(model_file)
        print(f"Loaded preprocessor and model")
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e.filename}")
        return

    # Apply feature engineering and transform
    X_test_fe = major_feature_engineering(X_test)
    X_test_processed = preprocessor.transform(X_test_fe)
    print(f"Processed test data shape: {X_test_processed.shape}")

    # Generate predictions and inverse transform
    log_predictions = model.predict(X_test_processed)
    final_predictions = np.expm1(log_predictions)
    final_predictions = np.maximum(final_predictions, 0)
    
    print("Predictions generated and inverse-transformed")

    # Create submission file
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'HotelValue': final_predictions
    })
    
    submission_df.to_csv(submission_filename, index=False)
    
    print(f"\n=== Submission Complete ===")
    print(f"File saved: '{submission_filename}'")
    print(f"\nFirst 5 predictions:")
    print(submission_df.head())
    print(f"\nPrediction statistics:")
    print(f"  Mean:   ${final_predictions.mean():,.2f}")
    print(f"  Median: ${np.median(final_predictions):,.2f}")
    print(f"  Min:    ${final_predictions.min():,.2f}")
    print(f"  Max:    ${final_predictions.max():,.2f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Step 1: Train the model
    model, preprocessor = train_and_evaluate_model()
    print("\n=== Training Complete ===")
    
    # Step 2: Generate predictions
    run_prediction(
        model_file="fitted_linear_model.joblib",
        submission_filename="submission_linear.csv"
    )
