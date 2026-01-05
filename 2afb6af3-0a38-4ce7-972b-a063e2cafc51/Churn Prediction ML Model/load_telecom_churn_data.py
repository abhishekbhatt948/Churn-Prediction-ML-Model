import pandas as pd
import numpy as np

# Generate a standard telecom customer churn dataset
np.random.seed(42)
n_customers = 7043

# Customer demographics
customer_id = range(1, n_customers + 1)
gender = np.random.choice(['Male', 'Female'], n_customers)
senior_citizen = np.random.choice([0, 1], n_customers, p=[0.84, 0.16])
partner = np.random.choice(['Yes', 'No'], n_customers)
dependents = np.random.choice(['Yes', 'No'], n_customers, p=[0.7, 0.3])

# Service usage and account information
tenure = np.random.randint(0, 73, n_customers)  # months
phone_service = np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1])
multiple_lines = np.where(
    phone_service == 'Yes',
    np.random.choice(['Yes', 'No', 'No phone service'], n_customers, p=[0.45, 0.45, 0.1]),
    'No phone service'
)

# Internet services
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.34, 0.44, 0.22])
online_security = np.where(
    internet_service != 'No',
    np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.35, 0.60, 0.05]),
    'No internet service'
)
online_backup = np.where(
    internet_service != 'No',
    np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.38, 0.57, 0.05]),
    'No internet service'
)
device_protection = np.where(
    internet_service != 'No',
    np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.36, 0.59, 0.05]),
    'No internet service'
)
tech_support = np.where(
    internet_service != 'No',
    np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.33, 0.62, 0.05]),
    'No internet service'
)
streaming_tv = np.where(
    internet_service != 'No',
    np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.42, 0.53, 0.05]),
    'No internet service'
)
streaming_movies = np.where(
    internet_service != 'No',
    np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.40, 0.55, 0.05]),
    'No internet service'
)

# Contract and billing
contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.55, 0.21, 0.24])
paperless_billing = np.random.choice(['Yes', 'No'], n_customers, p=[0.59, 0.41])
payment_method = np.random.choice([
    'Electronic check', 
    'Mailed check', 
    'Bank transfer (automatic)', 
    'Credit card (automatic)'
], n_customers, p=[0.34, 0.23, 0.22, 0.21])

# Charges - realistic pricing based on services
base_charge = np.where(internet_service == 'Fiber optic', 
                       np.random.uniform(75, 110, n_customers),
                       np.where(internet_service == 'DSL',
                               np.random.uniform(45, 75, n_customers),
                               np.random.uniform(20, 35, n_customers)))

# Tenure discount (longer tenure = slightly lower charges)
tenure_discount = np.where(tenure > 50, 0.9, np.where(tenure > 24, 0.95, 1.0))
monthly_charges = base_charge * tenure_discount

# Total charges = monthly charges * tenure (with some variation)
total_charges = monthly_charges * tenure * np.random.uniform(0.95, 1.05, n_customers)
# Set total charges to 0 for very new customers
total_charges = np.where(tenure == 0, 0, total_charges)

# Churn label - higher churn for month-to-month, fiber optic without support, short tenure
churn_prob = np.where(contract_type == 'Month-to-month', 0.42,
                     np.where(contract_type == 'One year', 0.11, 0.03))
churn_prob *= np.where(tenure < 6, 1.8, np.where(tenure < 24, 1.2, 1.0))
churn_prob *= np.where((internet_service == 'Fiber optic') & (tech_support == 'No'), 1.4, 1.0)
churn_prob *= np.where(senior_citizen == 1, 1.3, 1.0)
churn_prob = np.clip(churn_prob, 0, 0.7)

churn = np.random.binomial(1, churn_prob)
churn_label = np.where(churn == 1, 'Yes', 'No')

# Create DataFrame
telecom_churn_df = pd.DataFrame({
    'customerID': [f'CUST{str(i).zfill(4)}' for i in customer_id],
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract_type,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': np.round(monthly_charges, 2),
    'TotalCharges': np.round(total_charges, 2),
    'Churn': churn_label
})

# Display dataset information
print("=" * 70)
print("TELECOM CUSTOMER CHURN DATASET")
print("=" * 70)
print(f"\nDataset Shape: {telecom_churn_df.shape[0]} customers × {telecom_churn_df.shape[1]} features")
print(f"\nChurn Distribution:")
churn_counts = telecom_churn_df['Churn'].value_counts()
churn_pct = telecom_churn_df['Churn'].value_counts(normalize=True) * 100
for label in ['No', 'Yes']:
    print(f"  {label}: {churn_counts[label]:,} ({churn_pct[label]:.1f}%)")

print(f"\n{'Feature Categories'}")
print("-" * 70)
print(f"Demographics: gender, SeniorCitizen, Partner, Dependents")
print(f"Account Info: tenure, Contract, PaperlessBilling, PaymentMethod")
print(f"Services: PhoneService, MultipleLines, InternetService")
print(f"Add-ons: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,")
print(f"         StreamingTV, StreamingMovies")
print(f"Charges: MonthlyCharges, TotalCharges")
print(f"Target: Churn (binary)")

print(f"\n{'First 5 Rows'}")
print("-" * 70)
print(telecom_churn_df.head())

print(f"\n{'Data Types'}")
print("-" * 70)
print(telecom_churn_df.dtypes)

print(f"\n{'Statistical Summary'}")
print("-" * 70)
print(telecom_churn_df.describe())

print(f"\n✓ Dataset ready for churn prediction analysis")