import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Zerve design system
BG_COLOR = '#1D1D20'
TEXT_PRIMARY = '#fbfbff'
TEXT_SECONDARY = '#909094'
ZERVE_COLORS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']
HIGHLIGHT = '#ffd400'
SUCCESS = '#17b26a'
WARNING = '#f04438'

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, gb_pred_proba)
roc_auc_val = auc(fpr, tpr)

roc_fig, roc_ax = plt.subplots(figsize=(10, 8))
roc_fig.patch.set_facecolor(BG_COLOR)
roc_ax.set_facecolor(BG_COLOR)

roc_ax.plot(fpr, tpr, color=ZERVE_COLORS[0], lw=3, label=f'ROC Curve (AUC = {roc_auc_val:.3f})')
roc_ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, lw=2, linestyle='--', label='Random Classifier')
roc_ax.set_xlim([0.0, 1.0])
roc_ax.set_ylim([0.0, 1.05])
roc_ax.set_xlabel('False Positive Rate', fontsize=13, color=TEXT_PRIMARY, fontweight='bold')
roc_ax.set_ylabel('True Positive Rate', fontsize=13, color=TEXT_PRIMARY, fontweight='bold')
roc_ax.set_title('ROC Curve - Churn Prediction Model', fontsize=16, color=TEXT_PRIMARY, fontweight='bold', pad=20)
roc_ax.legend(loc='lower right', fontsize=12, facecolor=BG_COLOR, edgecolor=TEXT_SECONDARY, labelcolor=TEXT_PRIMARY)
roc_ax.tick_params(colors=TEXT_PRIMARY, labelsize=11)
roc_ax.grid(True, alpha=0.2, color=TEXT_SECONDARY, linestyle='-', linewidth=0.5)
for spine in roc_ax.spines.values():
    spine.set_edgecolor(TEXT_SECONDARY)
    spine.set_linewidth(1.5)
plt.tight_layout()
roc_curve_chart = roc_fig
print("✅ ROC Curve created")

# Confusion Matrix Visualization
conf_matrix_fig, conf_ax = plt.subplots(figsize=(10, 8))
conf_matrix_fig.patch.set_facecolor(BG_COLOR)
conf_ax.set_facecolor(BG_COLOR)

conf_data = np.array([[cm[0,0], cm[0,1]], 
                       [cm[1,0], cm[1,1]]])
im = conf_ax.imshow(conf_data, interpolation='nearest', cmap='Blues')
conf_ax.figure.colorbar(im, ax=conf_ax)

classes = ['No Churn', 'Churn']
tick_marks = np.arange(len(classes))
conf_ax.set_xticks(tick_marks)
conf_ax.set_yticks(tick_marks)
conf_ax.set_xticklabels(classes, fontsize=12, color=TEXT_PRIMARY, fontweight='bold')
conf_ax.set_yticklabels(classes, fontsize=12, color=TEXT_PRIMARY, fontweight='bold')

thresh = conf_data.max() / 2.
for i in range(conf_data.shape[0]):
    for j in range(conf_data.shape[1]):
        conf_ax.text(j, i, format(conf_data[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_data[i, j] > thresh else "black",
                    fontsize=20, fontweight='bold')

conf_ax.set_ylabel('Actual', fontsize=13, color=TEXT_PRIMARY, fontweight='bold')
conf_ax.set_xlabel('Predicted', fontsize=13, color=TEXT_PRIMARY, fontweight='bold')
conf_ax.set_title('Confusion Matrix', fontsize=16, color=TEXT_PRIMARY, fontweight='bold', pad=20)
plt.tight_layout()
confusion_matrix_chart = conf_matrix_fig
print("✅ Confusion Matrix visualization created")

# Feature Importance
feature_importance_vals = gb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance_vals
}).sort_values('importance', ascending=False).head(15)

feat_imp_fig, feat_ax = plt.subplots(figsize=(12, 8))
feat_imp_fig.patch.set_facecolor(BG_COLOR)
feat_ax.set_facecolor(BG_COLOR)

bars = feat_ax.barh(range(len(feature_importance_df)), feature_importance_df['importance'], 
                     color=ZERVE_COLORS[1], edgecolor=TEXT_PRIMARY, linewidth=1.5)
feat_ax.set_yticks(range(len(feature_importance_df)))
feat_ax.set_yticklabels(feature_importance_df['feature'], fontsize=11, color=TEXT_PRIMARY)
feat_ax.set_xlabel('Importance Score', fontsize=13, color=TEXT_PRIMARY, fontweight='bold')
feat_ax.set_title('Top 15 Feature Importances', fontsize=16, color=TEXT_PRIMARY, fontweight='bold', pad=20)
feat_ax.tick_params(colors=TEXT_PRIMARY, labelsize=11)
feat_ax.invert_yaxis()
for spine in feat_ax.spines.values():
    spine.set_edgecolor(TEXT_SECONDARY)
    spine.set_linewidth(1.5)
plt.tight_layout()
feature_importance_chart = feat_imp_fig
print("✅ Feature Importance chart created")

# Performance Metrics Bar Chart
metrics_fig, metrics_ax = plt.subplots(figsize=(10, 7))
metrics_fig.patch.set_facecolor(BG_COLOR)
metrics_ax.set_facecolor(BG_COLOR)

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [model_accuracy, model_precision, model_recall, model_f1, model_roc_auc]
colors_list = [ZERVE_COLORS[i % len(ZERVE_COLORS)] for i in range(len(metrics_names))]

bars = metrics_ax.bar(metrics_names, metrics_values, color=colors_list, 
                       edgecolor=TEXT_PRIMARY, linewidth=2, alpha=0.9)
metrics_ax.set_ylim([0, 1])
metrics_ax.set_ylabel('Score', fontsize=13, color=TEXT_PRIMARY, fontweight='bold')
metrics_ax.set_title('Model Performance Metrics', fontsize=16, color=TEXT_PRIMARY, fontweight='bold', pad=20)
metrics_ax.tick_params(colors=TEXT_PRIMARY, labelsize=11)

for i, (bar, value) in enumerate(zip(bars, metrics_values)):
    height = bar.get_height()
    metrics_ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}',
                    ha='center', va='bottom', color=TEXT_PRIMARY, fontsize=12, fontweight='bold')

for spine in metrics_ax.spines.values():
    spine.set_edgecolor(TEXT_SECONDARY)
    spine.set_linewidth(1.5)
plt.tight_layout()
performance_metrics_chart = metrics_fig
print("✅ Performance Metrics chart created")

print("\n" + "=" * 70)
print("✅ ALL VISUALIZATIONS COMPLETE")
print("=" * 70)
