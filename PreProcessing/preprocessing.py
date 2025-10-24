import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Define the path to the dataset folder
DATA_PATH = '../dataset/' # Go up one level (from preprocessing/) and then into dataset/

def major_feature_engineering(df):
    """
    Performs major feature engineering steps on the feature set, 
    creating new, aggregated features from existing ones.
    """
    df = df.copy()

    # --- 1. Area and Count Aggregations ---
    df['TotalSF'] = (df['GroundFloorArea'] + df['UpperFloorArea'] + df['BasementTotalSF'])
    df['OverallScore'] = (df['OverallQuality'] + df['OverallCondition']) / 2.0
    df['TotalBaths'] = (df['FullBaths'] + 0.5 * df['HalfBaths'] +
                        df['BasementFullBaths'] + 0.5 * df['BasementHalfBaths'])

    # --- 2. Temporal Features ---
    df['Age'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRemodel'] = df['YearSold'] - df['RenovationYear']
    # Handle negative values
    df['YearsSinceRemodel'] = np.where(df['YearsSinceRemodel'] < 0, 0, df['YearsSinceRemodel'])
    df.loc[df['RenovationYear'] == 0, 'YearsSinceRemodel'] = df['Age']

    # --- 3. Interaction Feature (Example) ---
    df['Qual_x_GroundSF'] = df['OverallQuality'] * df['GroundFloorArea']

    # --- 4. Feature Reduction/Drop ---
    # Drop original columns used for engineering to reduce multicollinearity
    drop_cols = ['GroundFloorArea', 'UpperFloorArea', 'BasementTotalSF',
                 'ConstructionYear', 'RenovationYear', 'OverallQuality',
                 'OverallCondition', 'FullBaths', 'HalfBaths',
                 'BasementFullBaths', 'BasementHalfBaths']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    return df

def remove_outliers(X, y):
    """
    Removes key outliers from the training data based on domain knowledge and IQR 
    on the total area, ensuring the model is not skewed by extreme, rare values.
    """
    data = pd.concat([X, y.rename('HotelValue')], axis=1)
    
    # Calculate TotalSF temporarily for IQR
    data['TotalSF'] = (data['GroundFloorArea'].fillna(0) + data['UpperFloorArea'].fillna(0) + data['BasementTotalSF'].fillna(0))
    
    # 1. Simple, high-leverage outlier removal (Large Land Area, Low Price)
    # Filter based on domain knowledge
    data = data.loc[~((data['LandArea'] > 50000) & (data['HotelValue'] < 200000))]

    # 2. IQR-based removal for combined area (TotalSF)
    q1 = data['TotalSF'].quantile(0.25)
    q3 = data['TotalSF'].quantile(0.75)
    iqr = q3 - q1
    # Remove observations with TotalSF greater than Q3 + 3*IQR
    data = data.loc[data['TotalSF'] <= (q3 + 3 * iqr)]
    
    # Drop temporary column
    data = data.drop(columns=['TotalSF'])
    
    return data.drop(columns=['HotelValue']), data['HotelValue']

def create_and_fit_preprocessor(X_train_fe):
    """
    Creates, fits, and saves the fitted ColumnTransformer for consistent
    imputation, scaling, and encoding.
    """
    numerical_features = X_train_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train_fe.select_dtypes(include=['object']).columns.tolist()
    
    # Numerical Transformer: Impute with median, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Median is robust to remaining outliers
        ('scaler', StandardScaler())
    ])

    # Categorical Transformer: Impute NaNs with 'None', then One-Hot Encode (captures absence of feature)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    preprocessor.fit(X_train_fe)
    joblib.dump(preprocessor, 'fitted_preprocessor.joblib')
    
    return preprocessor

if __name__ == '__main__':
    # --- Execution Block to Generate Joblib File ---
    try:
        # Correctly join the path to the file location
        train_df = pd.read_csv(DATA_PATH + 'train.csv')
        
        # 1. Separate
        y_full = train_df['HotelValue']
        X_full = train_df.drop(columns=['Id', 'HotelValue'])

        # 2. Outlier Removal
        X_full_clean, y_full_clean = remove_outliers(X_full, y_full)

        # 3. Feature Engineering
        X_full_fe = major_feature_engineering(X_full_clean)

        # 4. Fit and Save Preprocessor
        create_and_fit_preprocessor(X_full_fe)

        print("\n'preprocessing.py' executed successfully.")
        print("The necessary file 'fitted_preprocessor.joblib' is ready for model training.")
        
    except FileNotFoundError:
        print(f"ERROR: Could not find data files. Ensure 'train.csv' is in the '{DATA_PATH}' folder relative to this script.")
