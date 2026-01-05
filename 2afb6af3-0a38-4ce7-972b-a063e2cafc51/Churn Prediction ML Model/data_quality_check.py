import pandas as pd
import numpy as np

# Create a copy for processing
df_clean = telecom_churn_df.copy()

print("=" * 70)
print("DATA QUALITY ASSESSMENT")
print("=" * 70)

# 1. Missing Values Analysis
print("\n1. Missing Values Check:")
print("-" * 70)
missing_values = df_clean.isnull().sum()
missing_pct = (missing_values / len(df_clean)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if len(missing_df) > 0:
    print(missing_df)
else:
    print("✓ No missing values detected in any column")

# 2. Data Types Check
print("\n2. Data Types:")
print("-" * 70)
print(df_clean.dtypes)

# 3. Duplicate Records
print("\n3. Duplicate Records:")
print("-" * 70)
duplicates = df_clean.duplicated().sum()
print(f"Total duplicates: {duplicates}")
if duplicates > 0:
    df_clean = df_clean.drop_duplicates()
    print(f"✓ Removed {duplicates} duplicate records")
else:
    print("✓ No duplicate records found")

# 4. Unique Values in Categorical Columns
print("\n4. Categorical Features - Unique Values:")
print("-" * 70)
categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    unique_count = df_clean[col].nunique()
    print(f"{col:20s}: {unique_count:3d} unique values")
    if unique_count <= 10:
        print(f"  → Values: {sorted(df_clean[col].unique())}")

# 5. Numerical Features Summary
print("\n5. Numerical Features Statistics:")
print("-" * 70)
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
print(df_clean[numerical_cols].describe())

# 6. Check for unusual values in numerical columns
print("\n6. Data Range Validation:")
print("-" * 70)
print(f"Tenure range: {df_clean['tenure'].min()} to {df_clean['tenure'].max()} months")
print(f"MonthlyCharges range: ${df_clean['MonthlyCharges'].min():.2f} to ${df_clean['MonthlyCharges'].max():.2f}")
print(f"TotalCharges range: ${df_clean['TotalCharges'].min():.2f} to ${df_clean['TotalCharges'].max():.2f}")

# Check for negative values
negative_charges = (df_clean['MonthlyCharges'] < 0).sum() + (df_clean['TotalCharges'] < 0).sum()
if negative_charges > 0:
    print(f"⚠ Warning: {negative_charges} records with negative charges found")
else:
    print("✓ No negative charges detected")

# 7. Target Variable Distribution
print("\n7. Target Variable (Churn) Distribution:")
print("-" * 70)
churn_dist = df_clean['Churn'].value_counts()
churn_pct = df_clean['Churn'].value_counts(normalize=True) * 100
print(f"No:  {churn_dist['No']:,} ({churn_pct['No']:.2f}%)")
print(f"Yes: {churn_dist['Yes']:,} ({churn_pct['Yes']:.2f}%)")

# Calculate class imbalance ratio
imbalance_ratio = churn_dist['No'] / churn_dist['Yes']
print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1 (No:Yes)")
if imbalance_ratio > 3:
    print("⚠ Moderate class imbalance detected - consider stratified sampling or balancing techniques")

print("\n" + "=" * 70)
print("✓ Data Quality Check Complete")
print("=" * 70)
print(f"\nFinal Dataset: {df_clean.shape[0]} records × {df_clean.shape[1]} features")
