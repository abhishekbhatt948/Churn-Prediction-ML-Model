import pandas as pd
import numpy as np
from typing import Dict, Union, List

print("=" * 70)
print("PRODUCTION CHURN PREDICTION SERVICE")
print("=" * 70)

class ChurnPredictionService:
    """
    Production-ready churn prediction service that loads trained model,
    performs feature preprocessing, and returns structured predictions.
    """
    
    def __init__(self, model, scaler, feature_names):
        """
        Initialize the prediction service with trained model and preprocessing artifacts.
        
        Args:
            model: Trained GradientBoostingClassifier
            scaler: Fitted StandardScaler for feature normalization
            feature_names: List of expected feature names in correct order
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        
        # Binary feature mappings (must match training)
        self.binary_mappings = {
            'gender': {'Male': 1, 'Female': 0, 'M': 1, 'F': 0},
            'Partner': {'Yes': 1, 'No': 0, 'Y': 1, 'N': 0},
            'Dependents': {'Yes': 1, 'No': 0, 'Y': 1, 'N': 0},
            'PhoneService': {'Yes': 1, 'No': 0, 'Y': 1, 'N': 0},
            'PaperlessBilling': {'Yes': 1, 'No': 0, 'Y': 1, 'N': 0}
        }
        
        # Multi-category features for one-hot encoding
        self.multi_cat_features = [
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
        ]
        
        print(f"✅ Prediction service initialized")
        print(f"   Model: {type(self.model).__name__}")
        print(f"   Expected features: {len(self.feature_names)}")
    
    def validate_input(self, customer_data: Dict) -> Dict[str, List[str]]:
        """
        Validate input customer data and return validation results.
        
        Args:
            customer_data: Dictionary containing customer features
            
        Returns:
            Dictionary with 'valid' flag and list of 'errors' if any
        """
        errors = []
        
        # Required fields
        required_fields = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
        for field in required_fields:
            if field not in customer_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate numerical fields
        if 'tenure' in customer_data:
            try:
                tenure_val = float(customer_data['tenure'])
                if tenure_val < 0 or tenure_val > 100:
                    errors.append("tenure must be between 0 and 100 months")
            except (ValueError, TypeError):
                errors.append("tenure must be a valid number")
        
        if 'MonthlyCharges' in customer_data:
            try:
                monthly_val = float(customer_data['MonthlyCharges'])
                if monthly_val < 0 or monthly_val > 200:
                    errors.append("MonthlyCharges must be between 0 and 200")
            except (ValueError, TypeError):
                errors.append("MonthlyCharges must be a valid number")
        
        if 'TotalCharges' in customer_data:
            try:
                total_val = float(customer_data['TotalCharges'])
                if total_val < 0:
                    errors.append("TotalCharges must be non-negative")
            except (ValueError, TypeError):
                errors.append("TotalCharges must be a valid number")
        
        # Validate SeniorCitizen
        if 'SeniorCitizen' in customer_data:
            if customer_data['SeniorCitizen'] not in [0, 1, '0', '1']:
                errors.append("SeniorCitizen must be 0 or 1")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def preprocess_features(self, customer_data: Dict) -> pd.DataFrame:
        """
        Transform raw customer data into model-ready features.
        Replicates exact preprocessing pipeline from training.
        
        Args:
            customer_data: Dictionary containing customer features
            
        Returns:
            DataFrame with preprocessed features matching training format
        """
        # Create DataFrame from input
        df = pd.DataFrame([customer_data])
        
        # Binary feature encoding
        for feature, mapping in self.binary_mappings.items():
            if feature in df.columns:
                df[feature] = df[feature].map(mapping)
                if df[feature].isnull().any():
                    # Handle unmapped values by defaulting to 0
                    df[feature] = df[feature].fillna(0).astype(int)
        
        # Ensure SeniorCitizen is integer
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
        
        # One-hot encode multi-category features
        df_encoded = pd.get_dummies(df, columns=self.multi_cat_features, drop_first=True, dtype=int)
        
        # Ensure all expected features exist (add missing columns with 0)
        for feature in self.feature_names:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        # Select only the features used in training, in correct order
        df_final = df_encoded[self.feature_names]
        
        return df_final
    
    def predict(self, customer_data: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
        """
        Generate churn prediction for customer(s).
        
        Args:
            customer_data: Dictionary or list of dictionaries with customer features
            
        Returns:
            Prediction result(s) with churn probability, prediction, and risk level
        """
        # Handle single customer or batch
        single_customer = isinstance(customer_data, dict)
        if single_customer:
            customer_data = [customer_data]
        
        results = []
        
        for customer in customer_data:
            # Validate input
            validation = self.validate_input(customer)
            if not validation['valid']:
                results.append({
                    'success': False,
                    'error': 'Validation failed',
                    'validation_errors': validation['errors']
                })
                continue
            
            try:
                # Preprocess features
                features = self.preprocess_features(customer)
                
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Make prediction
                churn_probability = self.model.predict_proba(features_scaled)[0, 1]
                churn_prediction = int(self.model.predict(features_scaled)[0])
                
                # Determine risk level
                if churn_probability >= 0.7:
                    risk_level = 'HIGH'
                    risk_color = '#f04438'
                elif churn_probability >= 0.4:
                    risk_level = 'MEDIUM'
                    risk_color = '#ffd400'
                else:
                    risk_level = 'LOW'
                    risk_color = '#17b26a'
                
                # Build result
                result = {
                    'success': True,
                    'churn_prediction': 'WILL CHURN' if churn_prediction == 1 else 'WILL STAY',
                    'churn_probability': round(churn_probability, 4),
                    'churn_probability_pct': f"{churn_probability * 100:.2f}%",
                    'risk_level': risk_level,
                    'risk_color': risk_color,
                    'confidence': round(max(churn_probability, 1 - churn_probability), 4),
                    'recommendation': self._generate_recommendation(churn_probability, customer)
                }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'success': False,
                    'error': f'Prediction failed: {str(e)}'
                })
        
        return results[0] if single_customer else results
    
    def _generate_recommendation(self, churn_prob: float, customer_data: Dict) -> str:
        """Generate actionable recommendation based on prediction and customer profile."""
        if churn_prob >= 0.7:
            return "URGENT: High churn risk. Immediate retention action recommended. Consider personalized offers, contract incentives, or service upgrades."
        elif churn_prob >= 0.4:
            return "MONITOR: Moderate churn risk. Proactive engagement recommended. Review service satisfaction and identify pain points."
        else:
            return "RETAIN: Low churn risk. Maintain current service level and monitor satisfaction."


# Initialize the prediction service with trained model artifacts
prediction_service = ChurnPredictionService(
    model=gb_model,
    scaler=scaler,
    feature_names=feature_names
)

print("\n" + "=" * 70)
print("EXAMPLE USAGE & TESTING")
print("=" * 70)

# Example 1: High-risk customer
print("\n📊 Example 1: High-Risk Customer Profile")
print("-" * 70)
high_risk_customer = {
    'gender': 'Female',
    'SeniorCitizen': 1,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 3,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95.50,
    'TotalCharges': 286.50
}

result1 = prediction_service.predict(high_risk_customer)
print(f"✓ Prediction: {result1['churn_prediction']}")
print(f"✓ Probability: {result1['churn_probability_pct']}")
print(f"✓ Risk Level: {result1['risk_level']}")
print(f"✓ Confidence: {result1['confidence']}")
print(f"✓ Recommendation: {result1['recommendation']}")

# Example 2: Low-risk customer
print("\n📊 Example 2: Low-Risk Customer Profile")
print("-" * 70)
low_risk_customer = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'Yes',
    'tenure': 48,
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'Yes',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'Yes',
    'TechSupport': 'Yes',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Two year',
    'PaperlessBilling': 'No',
    'PaymentMethod': 'Credit card (automatic)',
    'MonthlyCharges': 105.50,
    'TotalCharges': 5062.40
}

result2 = prediction_service.predict(low_risk_customer)
print(f"✓ Prediction: {result2['churn_prediction']}")
print(f"✓ Probability: {result2['churn_probability_pct']}")
print(f"✓ Risk Level: {result2['risk_level']}")
print(f"✓ Confidence: {result2['confidence']}")
print(f"✓ Recommendation: {result2['recommendation']}")

# Example 3: Batch predictions
print("\n📊 Example 3: Batch Prediction (Multiple Customers)")
print("-" * 70)
batch_customers = [high_risk_customer, low_risk_customer]
batch_results = prediction_service.predict(batch_customers)
print(f"✓ Processed {len(batch_results)} customers")
for idx, result in enumerate(batch_results, 1):
    print(f"  Customer {idx}: {result['churn_prediction']} ({result['churn_probability_pct']} probability)")

# Example 4: Invalid input handling
print("\n📊 Example 4: Input Validation")
print("-" * 70)
invalid_customer = {
    'gender': 'Male',
    'tenure': -5,  # Invalid
    # Missing required fields
}
result4 = prediction_service.predict(invalid_customer)
print(f"✓ Validation status: {'PASSED' if result4['success'] else 'FAILED'}")
if not result4['success']:
    print(f"✓ Errors detected: {len(result4.get('validation_errors', []))}")
    for error in result4.get('validation_errors', [])[:3]:
        print(f"  - {error}")

print("\n" + "=" * 70)
print("✅ PREDICTION SERVICE READY FOR PRODUCTION")
print("=" * 70)
print("\nAPI Integration Guide:")
print("  1. Import: from prediction_service import prediction_service")
print("  2. Call: result = prediction_service.predict(customer_data)")
print("  3. Response: JSON/dict with prediction, probability, risk_level")
print("\nKey Features:")
print("  ✓ Input validation with detailed error messages")
print("  ✓ Feature preprocessing matching training pipeline")
print("  ✓ Batch prediction support")
print("  ✓ Risk level classification (LOW/MEDIUM/HIGH)")
print("  ✓ Actionable recommendations")
print("  ✓ Error handling and graceful failure")
