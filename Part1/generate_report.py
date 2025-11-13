# %% [markdown]
# # Hotel Value Prediction - Exploratory Data Analysis
# This notebook performs comprehensive EDA on the hotel value dataset

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# %%
# Reading train data file
train_df = pd.read_csv("train.csv")
print("Train data loaded successfully!")
print(f"Shape: {train_df.shape}")

# %%
# Display first few rows
train_df.head()

# %%
# Basic information about the dataset
train_df.info()

# %%
# Check for null values
print("Missing Values Summary:")
print("="*50)
missing_summary = train_df.isnull().sum()
missing_pct = (train_df.isnull().sum() / len(train_df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_summary,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))

# %%
# Check for duplicates
duplicates = train_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
if duplicates > 0:
    train_df.drop_duplicates(inplace=True)
    print(f"Duplicates removed. New shape: {train_df.shape}")

# %%
# Statistical summary of numerical features
print("Statistical Summary of Numerical Features:")
print("="*80)
print(train_df.describe().T)

# %%
# Missing data visualization
print("Missing Data Visualization:")
msno.matrix(train_df, figsize=(14, 8))
plt.title('Missing Data Matrix', fontsize=16, pad=20)
plt.show()

# %%
# Identify numerical and categorical columns
num_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()

# Remove ID column from analysis if present
if 'hotel_id' in num_cols:
    num_cols.remove('hotel_id')
if 'hotel_id' in train_df.columns:
    print(f"ID column found: hotel_id")

print(f"\nNumerical columns ({len(num_cols)}): {num_cols}")
print(f"\nCategorical columns ({len(cat_cols)}): {cat_cols}")

# %% [markdown]
# ## Distribution Analysis of Numerical Features

# %%
# Distribution plots for numerical columns
for col in num_cols:
    plt.figure(figsize=(14, 5))
    
    # Histogram with KDE
    plt.subplot(1, 2, 1)
    sns.histplot(train_df[col].dropna(), kde=True, bins=30, color='steelblue')
    plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
    # Q-Q plot for normality check
    plt.subplot(1, 2, 2)
    stats.probplot(train_df[col].dropna(), dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {col}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print skewness and kurtosis
    skewness = train_df[col].skew()
    kurtosis_val = train_df[col].kurtosis()
    print(f"{col} - Skewness: {skewness:.3f}, Kurtosis: {kurtosis_val:.3f}")
    print("-" * 60)

# %% [markdown]
# ## Outlier Detection

# %%
# Boxplots for outlier detection
for col in num_cols:
    plt.figure(figsize=(12, 4))
    sns.boxplot(x=train_df[col], color='lightcoral')
    plt.title(f'Boxplot of {col} - Outlier Detection', fontsize=14, fontweight='bold')
    plt.xlabel(col)
    
    # Calculate outlier statistics
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)][col]
    
    plt.text(0.02, 0.98, f'Outliers: {len(outliers)} ({len(outliers)/len(train_df)*100:.2f}%)', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Correlation Analysis

# %%
# Correlation matrix
plt.figure(figsize=(14, 10))
correlation_matrix = train_df[num_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, mask=mask, linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numerical Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# %%
# Find highly correlated features
threshold = 0.7
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            high_corr.append({
                'Feature 1': correlation_matrix.columns[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

if high_corr:
    print("\nHighly Correlated Features (|correlation| > 0.7):")
    print("="*60)
    high_corr_df = pd.DataFrame(high_corr)
    print(high_corr_df.to_string(index=False))
else:
    print("\nNo highly correlated feature pairs found (threshold = 0.7)")

# %% [markdown]
# ## Categorical Features Analysis

# %%
# Analysis of categorical features
for col in cat_cols:
    if col == 'hotel_id':  # Skip ID column
        continue
    
    plt.figure(figsize=(14, 5))
    
    # Value counts
    value_counts = train_df[col].value_counts()
    
    # Bar plot
    plt.subplot(1, 2, 1)
    sns.countplot(data=train_df, y=col, order=value_counts.index, palette='viridis')
    plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
    plt.xlabel('Count')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=sns.color_palette('viridis', len(value_counts)))
    plt.title(f'Proportion of {col}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{col} - Unique values: {train_df[col].nunique()}")
    print(value_counts)
    print("-" * 60)

# %% [markdown]
# ## Target Variable Analysis (if present)

# %%
# Assuming target variable might be named 'hotel_value', 'price', 'value', etc.
target_candidates = ['hotel_value', 'price', 'value', 'target']
target_col = None

for candidate in target_candidates:
    if candidate in train_df.columns:
        target_col = candidate
        break

if target_col:
    print(f"Target variable found: {target_col}")
    
    plt.figure(figsize=(14, 10))
    
    # Original distribution
    plt.subplot(2, 2, 1)
    sns.histplot(train_df[target_col].dropna(), kde=True, bins=50, color='steelblue')
    plt.title(f'Original Distribution of {target_col}', fontsize=12, fontweight='bold')
    plt.xlabel(target_col)
    
    # Log-transformed distribution
    plt.subplot(2, 2, 2)
    log_target = np.log1p(train_df[target_col].dropna())
    sns.histplot(log_target, kde=True, bins=50, color='coral')
    plt.title(f'Log-Transformed Distribution of {target_col}', fontsize=12, fontweight='bold')
    plt.xlabel(f'log({target_col})')
    
    # Boxplot
    plt.subplot(2, 2, 3)
    sns.boxplot(x=train_df[target_col], color='lightgreen')
    plt.title(f'Boxplot of {target_col}', fontsize=12, fontweight='bold')
    
    # Q-Q plot
    plt.subplot(2, 2, 4)
    stats.probplot(train_df[target_col].dropna(), dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {target_col}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTarget Variable Statistics:")
    print(f"Mean: {train_df[target_col].mean():.2f}")
    print(f"Median: {train_df[target_col].median():.2f}")
    print(f"Std: {train_df[target_col].std():.2f}")
    print(f"Skewness: {train_df[target_col].skew():.3f}")
    print(f"Kurtosis: {train_df[target_col].kurtosis():.3f}")

# %% [markdown]
# ## Bivariate Analysis - Numerical Features vs Target

# %%
if target_col and target_col in num_cols:
    other_num_cols = [col for col in num_cols if col != target_col]
    
    for col in other_num_cols[:5]:  # Plot first 5 to avoid too many plots
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=train_df, x=col, y=target_col, alpha=0.5, color='steelblue')
        plt.title(f'{col} vs {target_col}', fontsize=14, fontweight='bold')
        plt.xlabel(col)
        plt.ylabel(target_col)
        
        # Add correlation coefficient
        corr = train_df[[col, target_col]].corr().iloc[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## Data Cleaning Summary

# %%
# Drop ID column if present
if 'hotel_id' in train_df.columns:
    train_df.drop('hotel_id', axis=1, inplace=True)
    print("Dropped 'hotel_id' column")

print("\nFinal Dataset Shape:", train_df.shape)
print("\nFinal Dataset Info:")
train_df.info()

# %% [markdown]
# ## Save Cleaned Data (Optional)

# %%
# Uncomment to save cleaned data
# train_df.to_csv('train_cleaned.csv', index=False)
# print("Cleaned data saved to 'train_cleaned.csv'")

print("\n" + "="*80)
print("EDA Complete!")
print("="*80)
