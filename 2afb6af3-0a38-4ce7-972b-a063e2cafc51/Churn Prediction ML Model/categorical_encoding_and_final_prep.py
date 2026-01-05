import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("=" * 70)
print("CATEGORICAL FEATURE ENCODING & FINAL DATA PREPARATION")
print("=" * 70)

# Create final modeling dataset
df_model = df_clean.copy()

# Drop customerID as it's not a feature
df_model = df_model.drop(['customerID', 'TenureGroup'], axis=1, errors='ignore')

print("\n1. Encoding Strategy:")
print("-" * 70)

# Binary categorical features - simple mapping
binary_features = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'Churn': {'Yes': 1, 'No': 0}
}

print("\nBinary Features (Label Encoding):")
for feature, mapping in binary_features.items():
    df_model[feature] = df_model[feature].map(mapping)
    print(f"  • {feature}: {list(mapping.keys())} → {list(mapping.values())}")

# Multi-level categorical features - one-hot encoding
multi_cat_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

print("\nMulti-Category Features (One-Hot Encoding):")
for feature in multi_cat_features:
    unique_vals = df_clean[feature].unique()
    print(f"  • {feature}: {len(unique_vals)} categories → {len(unique_vals)} dummy variables")

# Create dummy variables with drop_first=True to avoid multicollinearity
df_encoded = pd.get_dummies(df_model, columns=multi_cat_features, drop_first=True, dtype=int)

print(f"\n2. Final Dataset Dimensions:")
print("-" * 70)
print(f"Original features: {df_model.shape[1]}")
print(f"After one-hot encoding: {df_encoded.shape[1]}")
print(f"Total samples: {df_encoded.shape[0]}")

# Split features and target
X_features = df_encoded.drop('Churn', axis=1)
y_target = df_encoded['Churn']

print(f"\n3. Train-Ready Dataset:")
print("-" * 70)
print(f"Feature matrix (X): {X_features.shape[0]} samples × {X_features.shape[1]} features")
print(f"Target variable (y): {y_target.shape[0]} samples")
print(f"\nClass distribution:")
print(f"  • Class 0 (Not Churned): {(y_target == 0).sum():,} ({(y_target == 0).mean() * 100:.1f}%)")
print(f"  • Class 1 (Churned): {(y_target == 1).sum():,} ({(y_target == 1).mean() * 100:.1f}%)")

# Check for any remaining missing values
missing_check = X_features.isnull().sum().sum()
print(f"\n4. Data Quality Verification:")
print("-" * 70)
print(f"Missing values in feature matrix: {missing_check}")
print(f"Missing values in target: {y_target.isnull().sum()}")
print(f"Data types consistent: {(X_features.dtypes.isin(['int64', 'float64', 'int32'])).all()}")

# Feature list for modeling
feature_names = X_features.columns.tolist()
print(f"\n5. Feature Names ({len(feature_names)} total):")
print("-" * 70)
print(f"Numerical features: tenure, MonthlyCharges, TotalCharges, SeniorCitizen")
print(f"Binary encoded: gender, Partner, Dependents, PhoneService, PaperlessBilling")
print(f"One-hot encoded: {len(feature_names) - 9} features from multi-category variables")

# Summary statistics of encoded features
print(f"\n6. Encoded Feature Summary:")
print("-" * 70)
print(X_features.describe())

print("\n" + "=" * 70)
print("✓ Dataset Ready for Machine Learning Modeling")
print("=" * 70)
print("\nNext Steps:")
print("  1. Train-test split (stratified by churn)")
print("  2. Feature scaling for numerical variables")
print("  3. Model training (Logistic Regression, Random Forest, XGBoost, etc.)")
print("  4. Model evaluation and feature importance analysis")
print("  5. Hyperparameter tuning and cross-validation")
