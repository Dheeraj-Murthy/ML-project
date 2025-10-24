import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import joblib

warnings.filterwarnings('ignore')

def major_feature_engineering(df):
    """
    Performs major feature engineering steps for the Hotel Value dataset,
    creating new features that capture non-linear relationships and combining
    related features into more informative single variables.
    """
    df = df.copy()

    # --- 1. Area Features ---
    # Total Usable Area (Above Ground + Basement)
    df['TotalSF'] = (df['GroundFloorArea'] + df['UpperFloorArea'] + df['BasementTotalSF'])

    # --- 2. Temporal Features ---
    # Age of the property
    df['Age'] = df['YearSold'] - df['ConstructionYear']
    # Years since the last remodel/renovation
    df['YearsSinceRemodel'] = df['YearSold'] - df['RenovationYear']
    # Handle negative values
    df['YearsSinceRemodel'] = np.where(df['YearsSinceRemodel'] < 0, 0, df['YearsSinceRemodel'])
    # If no RenovationYear, use Age
    df.loc[df['RenovationYear'] == 0, 'YearsSinceRemodel'] = df['Age']

    # --- 3. Quality and Condition Scores ---
    # Simplified overall score
    df['OverallScore'] = (df['OverallQuality'] + df['OverallCondition']) / 2.0

    # --- 4. Count Features ---
    # Total Bathrooms (Full = 1, Half = 0.5)
    df['TotalBaths'] = (df['FullBaths'] + 0.5 * df['HalfBaths'] +
                        df['BasementFullBaths'] + 0.5 * df['BasementHalfBaths'])

    # --- 5. Interaction Features (Example) ---
    df['Qual_x_GroundSF'] = df['OverallQuality'] * df['GroundFloorArea']

    # --- 6. Feature Reduction/Drop ---
    drop_cols = ['GroundFloorArea', 'UpperFloorArea', 'BasementTotalSF',
                 'ConstructionYear', 'RenovationYear', 'OverallQuality',
                 'OverallCondition', 'FullBaths', 'HalfBaths',
                 'BasementFullBaths', 'BasementHalfBaths']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    return df

def create_preprocessor_pipeline(X_train):
    """
    Creates and fits the ColumnTransformer preprocessing pipeline based on the training data.
    
    Categorical NaNs are imputed with 'None' to capture absence of a feature.
    Numerical NaNs are imputed with the median, which is robust to outliers.
    """
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Numerical Transformer: Impute with median, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Transformer: Impute NaNs with 'None', then One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create the full preprocessor
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
    Main function to load, engineer, and preprocess the data.
    
    Returns:
        X_train_processed, X_test_processed, y_train_transformed, test_ids, preprocessor
    """
    # Load the datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    test_ids = test_df['Id']

    # Separate features and target
    y_full_train = train_df['HotelValue']
    X_full_train = train_df.drop(['Id', 'HotelValue'], axis=1)
    X_test = test_df.drop('Id', axis=1)

    # Apply target transformation
    y_train_transformed = np.log1p(y_full_train) if target_log_transform else y_full_train

    # --- Apply Feature Engineering ---
    X_full_train_fe = major_feature_engineering(X_full_train)
    X_test_fe = major_feature_engineering(X_test)

    # Ensure test set columns align with the train set columns after FE
    X_test_fe = X_test_fe[X_full_train_fe.columns]

    # --- Create and Fit Preprocessor ---
    preprocessor = create_preprocessor_pipeline(X_full_train_fe)
    
    # Fit the preprocessor ONLY on the training data
    preprocessor.fit(X_full_train_fe)

    # Save the fitted preprocessor for later use in prediction
    joblib.dump(preprocessor, 'fitted_preprocessor.joblib')

    # --- Transform Data ---
    X_train_processed = preprocessor.transform(X_full_train_fe)
    X_test_processed = preprocessor.transform(X_test_fe)

    print("Data Preprocessing and Feature Engineering Complete.")
    print(f"Fitted preprocessor saved to 'fitted_preprocessor.joblib'.")
    print(f"Shape of Processed Training Data: {X_train_processed.shape}")

    return X_train_processed, X_test_processed, y_train_transformed, test_ids, preprocessor

if __name__ == '__main__':
    # Execute the preprocessing script to get the ready data
    X_train_proc, X_test_proc, y_train_trans, test_ids, fitted_preprocessor = run_preprocessing(target_log_transform=True)
    
    print("\nSuccessfully loaded and transformed data. Ready for model training.")
