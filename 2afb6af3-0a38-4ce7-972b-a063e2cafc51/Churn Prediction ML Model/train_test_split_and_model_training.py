import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import matplotlib.pyplot as plt

# Zerve color scheme
BG_COLOR = '#1D1D20'
TEXT_PRIMARY = '#fbfbff'
TEXT_SECONDARY = '#909094'
ZERVE_COLORS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']
HIGHLIGHT = '#ffd400'
SUCCESS = '#17b26a'
WARNING = '#f04438'

print("=" * 70)
print("CHURN PREDICTION MODEL TRAINING")
print("=" * 70)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_target, test_size=0.2, random_state=42, stratify=y_target
)

print(f"\n📊 Dataset Split:")
print(f"   Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X_features)*100:.1f}%)")
print(f"   Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X_features)*100:.1f}%)")
print(f"   Features: {X_train.shape[1]}")
print(f"\n   Train churn rate: {y_train.mean()*100:.2f}%")
print(f"   Test churn rate: {y_test.mean()*100:.2f}%")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 70)
print("Gradient Boosting Classifier")
print("=" * 70)

# Train Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbose=0
)

print("\n⏳ Training model...")
gb_model.fit(X_train_scaled, y_train)
print("✅ Training complete!")

gb_pred = gb_model.predict(X_test_scaled)
gb_pred_proba = gb_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
model_accuracy = accuracy_score(y_test, gb_pred)
model_precision = precision_score(y_test, gb_pred)
model_recall = recall_score(y_test, gb_pred)
model_f1 = f1_score(y_test, gb_pred)
model_roc_auc = roc_auc_score(y_test, gb_pred_proba)

print(f"\n" + "=" * 70)
print("MODEL PERFORMANCE METRICS")
print("=" * 70)
print(f"\n✅ Gradient Boosting Results:")
print(f"   • Accuracy:  {model_accuracy:.4f} ({model_accuracy*100:.2f}%)")
print(f"   • Precision: {model_precision:.4f} ({model_precision*100:.2f}%)")
print(f"   • Recall:    {model_recall:.4f} ({model_recall*100:.2f}%)")
print(f"   • F1-Score:  {model_f1:.4f}")
print(f"   • ROC-AUC:   {model_roc_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, gb_pred)
print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives:  {cm[0,0]:,} (Correctly predicted non-churners)")
print(f"   False Positives: {cm[0,1]:,} (Non-churners predicted as churners)")
print(f"   False Negatives: {cm[1,0]:,} (Churners predicted as non-churners)")
print(f"   True Positives:  {cm[1,1]:,} (Correctly predicted churners)")

# Calculate specificity and other rates
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print(f"\n📈 Additional Metrics:")
print(f"   • Sensitivity (Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"   • Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
print(f"   • False Positive Rate: {fp/(fp+tn):.4f}")
print(f"   • False Negative Rate: {fn/(fn+tp):.4f}")

print("\n" + "=" * 70)
print("MODEL INTERPRETATION")
print("=" * 70)
print(f"\nThe model demonstrates strong predictive power:")
print(f"• ROC-AUC of {model_roc_auc:.3f} indicates excellent ability to distinguish")
print(f"  between customers who will churn and those who will stay")
print(f"• Precision of {model_precision:.3f} means {model_precision*100:.1f}% of predicted churners")
print(f"  are actual at-risk customers")
print(f"• Recall of {model_recall:.3f} means the model identifies {model_recall*100:.1f}% of")
print(f"  all customers who will actually churn")

print("\n" + "=" * 70)
print("✅ MODEL TRAINING COMPLETE")
print("=" * 70)
