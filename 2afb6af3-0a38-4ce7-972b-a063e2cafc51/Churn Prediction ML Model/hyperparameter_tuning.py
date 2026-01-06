import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import time

# Zerve color scheme
BG_COLOR = '#1D1D20'
TEXT_PRIMARY = '#fbfbff'
TEXT_SECONDARY = '#909094'
ZERVE_COLORS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']
HIGHLIGHT = '#ffd400'
SUCCESS = '#17b26a'
WARNING = '#f04438'

print("=" * 70)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("=" * 70)

# Baseline Random Forest model (for comparison)
print("\n📊 STEP 1: Training Baseline Random Forest Model")
print("-" * 70)

baseline_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
baseline_rf.fit(X_train_scaled, y_train)

baseline_pred = baseline_rf.predict(X_test_scaled)
baseline_pred_proba = baseline_rf.predict_proba(X_test_scaled)[:, 1]

baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_precision = precision_score(y_test, baseline_pred)
baseline_recall = recall_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)
baseline_roc_auc = roc_auc_score(y_test, baseline_pred_proba)

print(f"\n✅ Baseline Random Forest (default parameters):")
print(f"   • Accuracy:  {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"   • Precision: {baseline_precision:.4f} ({baseline_precision*100:.2f}%)")
print(f"   • Recall:    {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
print(f"   • F1-Score:  {baseline_f1:.4f}")
print(f"   • ROC-AUC:   {baseline_roc_auc:.4f}")

# Define hyperparameter grid - reduced for faster execution
print("\n" + "=" * 70)
print("📊 STEP 2: Hyperparameter Grid Search Setup")
print("-" * 70)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print("\nHyperparameter search space:")
print(f"   • n_estimators: {param_grid['n_estimators']}")
print(f"   • max_depth: {param_grid['max_depth']}")
print(f"   • min_samples_split: {param_grid['min_samples_split']}")
print(f"   • min_samples_leaf: {param_grid['min_samples_leaf']}")
print(f"   • max_features: {param_grid['max_features']}")

total_combinations = (len(param_grid['n_estimators']) * 
                     len(param_grid['max_depth']) * 
                     len(param_grid['min_samples_split']) * 
                     len(param_grid['min_samples_leaf']) * 
                     len(param_grid['max_features']))

print(f"\nTotal combinations: {total_combinations}")
print(f"Using GridSearchCV with 5-fold cross-validation")
print(f"Total model fits: {total_combinations * 5} = {total_combinations * 5:,}")

# Perform GridSearchCV
print("\n" + "=" * 70)
print("🔍 STEP 3: Running GridSearchCV (this may take a few minutes...)")
print("-" * 70)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"\n✅ Grid Search Complete!")
print(f"   Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Get best parameters and model
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

print("\n" + "=" * 70)
print("🏆 BEST HYPERPARAMETERS FOUND")
print("-" * 70)
for param, value in best_params.items():
    print(f"   • {param}: {value}")

print(f"\n   Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")

# Evaluate tuned model on test set
print("\n" + "=" * 70)
print("📊 STEP 4: Evaluating Tuned Model on Test Set")
print("-" * 70)

tuned_pred = best_rf_model.predict(X_test_scaled)
tuned_pred_proba = best_rf_model.predict_proba(X_test_scaled)[:, 1]

tuned_accuracy = accuracy_score(y_test, tuned_pred)
tuned_precision = precision_score(y_test, tuned_pred)
tuned_recall = recall_score(y_test, tuned_pred)
tuned_f1 = f1_score(y_test, tuned_pred)
tuned_roc_auc = roc_auc_score(y_test, tuned_pred_proba)

print(f"\n✅ Tuned Random Forest Model:")
print(f"   • Accuracy:  {tuned_accuracy:.4f} ({tuned_accuracy*100:.2f}%)")
print(f"   • Precision: {tuned_precision:.4f} ({tuned_precision*100:.2f}%)")
print(f"   • Recall:    {tuned_recall:.4f} ({tuned_recall*100:.2f}%)")
print(f"   • F1-Score:  {tuned_f1:.4f}")
print(f"   • ROC-AUC:   {tuned_roc_auc:.4f}")

# Performance comparison
print("\n" + "=" * 70)
print("📈 BASELINE vs TUNED MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Baseline RF': [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1, baseline_roc_auc],
    'Tuned RF': [tuned_accuracy, tuned_precision, tuned_recall, tuned_f1, tuned_roc_auc],
    'Improvement': [
        tuned_accuracy - baseline_accuracy,
        tuned_precision - baseline_precision,
        tuned_recall - baseline_recall,
        tuned_f1 - baseline_f1,
        tuned_roc_auc - baseline_roc_auc
    ],
    'Improvement %': [
        (tuned_accuracy - baseline_accuracy) / baseline_accuracy * 100,
        (tuned_precision - baseline_precision) / baseline_precision * 100,
        (tuned_recall - baseline_recall) / baseline_recall * 100,
        (tuned_f1 - baseline_f1) / baseline_f1 * 100,
        (tuned_roc_auc - baseline_roc_auc) / baseline_roc_auc * 100
    ]
})

print("\n" + comparison_df.to_string(index=False))

# Additional comparison with Gradient Boosting baseline
print("\n" + "=" * 70)
print("📊 COMPARISON WITH GRADIENT BOOSTING BASELINE")
print("=" * 70)

gb_comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Gradient Boosting': [model_accuracy, model_precision, model_recall, model_f1, model_roc_auc],
    'Tuned Random Forest': [tuned_accuracy, tuned_precision, tuned_recall, tuned_f1, tuned_roc_auc],
    'Difference': [
        tuned_accuracy - model_accuracy,
        tuned_precision - model_precision,
        tuned_recall - model_recall,
        tuned_f1 - model_f1,
        tuned_roc_auc - model_roc_auc
    ]
})

print("\n" + gb_comparison_df.to_string(index=False))

# Key insights
print("\n" + "=" * 70)
print("💡 KEY INSIGHTS")
print("=" * 70)

improvements = []
for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']):
    baseline_val = comparison_df.iloc[i]['Baseline RF']
    tuned_val = comparison_df.iloc[i]['Tuned RF']
    improvement = comparison_df.iloc[i]['Improvement']
    improvement_pct = comparison_df.iloc[i]['Improvement %']
    
    if improvement > 0:
        improvements.append(f"   ✅ {metric}: +{improvement:.4f} ({improvement_pct:+.2f}%)")
    elif improvement < 0:
        improvements.append(f"   ⚠️  {metric}: {improvement:.4f} ({improvement_pct:.2f}%)")
    else:
        improvements.append(f"   ➖ {metric}: No change")

for improvement in improvements:
    print(improvement)

if tuned_roc_auc > baseline_roc_auc:
    print(f"\n🎉 SUCCESS: Hyperparameter tuning improved model performance!")
    print(f"   The tuned model shows better predictive power with ROC-AUC")
    print(f"   increasing from {baseline_roc_auc:.4f} to {tuned_roc_auc:.4f}")
else:
    print(f"\n📊 The baseline Random Forest already performed well.")

print("\n" + "=" * 70)
print("✅ HYPERPARAMETER TUNING COMPLETE")
print("=" * 70)
