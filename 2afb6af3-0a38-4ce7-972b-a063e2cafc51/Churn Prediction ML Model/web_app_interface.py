import pandas as pd
import matplotlib.pyplot as plt

# Zerve design colors
BG_COLOR = '#1D1D20'
TEXT_PRIMARY = '#fbfbff'
TEXT_SECONDARY = '#909094'
ZERVE_COLORS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']
HIGHLIGHT = '#ffd400'
SUCCESS = '#17b26a'
WARNING = '#f04438'

print("=" * 80)
print("🎯 CHURN PREDICTION WEB APPLICATION")
print("=" * 80)
print("\n📝 Example Customer Scenarios & API Integration Demo")
print("=" * 80)

# Test Scenario 1: High-Risk Customer
print("\n" + "─" * 80)
print("SCENARIO 1: HIGH-RISK CUSTOMER")
print("─" * 80)

high_risk_customer_web = {
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

print("\\nCustomer Profile:")
print(f"  • Tenure: {high_risk_customer_web['tenure']} months")
print(f"  • Contract: {high_risk_customer_web['Contract']}")
print(f"  • Internet: {high_risk_customer_web['InternetService']}")
print(f"  • Tech Support: {high_risk_customer_web['TechSupport']}")
print(f"  • Monthly Charges: ${high_risk_customer_web['MonthlyCharges']}")

# Call prediction service
result_hr = prediction_service.predict(high_risk_customer_web)

print("\\n🔮 Prediction Results:")
print(f"  • Churn Prediction: {result_hr['churn_prediction']}")
print(f"  • Churn Probability: {result_hr['churn_probability_pct']} ({result_hr['churn_probability']})")
print(f"  • Risk Level: {result_hr['risk_level']}")
print(f"  • Model Confidence: {result_hr['confidence']}")
print(f"\\n💡 Recommendation: {result_hr['recommendation']}")

# Test Scenario 2: Low-Risk Customer
print("\\n" + "─" * 80)
print("SCENARIO 2: LOW-RISK CUSTOMER")
print("─" * 80)

low_risk_customer_web = {
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

print("\\nCustomer Profile:")
print(f"  • Tenure: {low_risk_customer_web['tenure']} months")
print(f"  • Contract: {low_risk_customer_web['Contract']}")
print(f"  • Internet: {low_risk_customer_web['InternetService']}")
print(f"  • Tech Support: {low_risk_customer_web['TechSupport']}")
print(f"  • Monthly Charges: ${low_risk_customer_web['MonthlyCharges']}")

# Call prediction service
result_lr = prediction_service.predict(low_risk_customer_web)

print("\\n🔮 Prediction Results:")
print(f"  • Churn Prediction: {result_lr['churn_prediction']}")
print(f"  • Churn Probability: {result_lr['churn_probability_pct']} ({result_lr['churn_probability']})")
print(f"  • Risk Level: {result_lr['risk_level']}")
print(f"  • Model Confidence: {result_lr['confidence']}")
print(f"\\n💡 Recommendation: {result_lr['recommendation']}")

# Comprehensive Visualizations
print("\\n" + "=" * 80)
print("📊 COMPREHENSIVE RESULTS VISUALIZATION")
print("=" * 80)

fig_combined = plt.figure(figsize=(16, 10))
fig_combined.patch.set_facecolor(BG_COLOR)

# Scenario 1 visualizations
ax1 = plt.subplot(2, 3, 1)
ax1.set_facecolor(BG_COLOR)

# High-risk probability bar
categories_hr = ['Will Stay', 'Will Churn']
probabilities_hr = [1 - result_hr['churn_probability'], result_hr['churn_probability']]
colors_hr = [ZERVE_COLORS[2], ZERVE_COLORS[3]]

bars_hr = ax1.bar(categories_hr, probabilities_hr, color=colors_hr, width=0.6)

for bar_hr, value_hr in zip(bars_hr, probabilities_hr):
    height_hr = bar_hr.get_height()
    ax1.text(bar_hr.get_x() + bar_hr.get_width()/2., height_hr + 0.02,
            f'{value_hr*100:.1f}%', ha='center', va='bottom', color=TEXT_PRIMARY, 
            fontsize=10, fontweight='bold')

ax1.set_ylabel('Probability', fontsize=11, color=TEXT_PRIMARY, fontweight='bold')
ax1.set_title('Scenario 1: High-Risk Customer\\nChurn Probability', fontsize=12, 
             color=TEXT_PRIMARY, fontweight='bold', pad=15)
ax1.set_ylim(0, 1.1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color(TEXT_SECONDARY)
ax1.spines['bottom'].set_color(TEXT_SECONDARY)
ax1.tick_params(colors=TEXT_PRIMARY, labelsize=9)

# Scenario 2 visualizations
ax2 = plt.subplot(2, 3, 2)
ax2.set_facecolor(BG_COLOR)

# Low-risk probability bar
categories_lr = ['Will Stay', 'Will Churn']
probabilities_lr = [1 - result_lr['churn_probability'], result_lr['churn_probability']]
colors_lr = [ZERVE_COLORS[2], ZERVE_COLORS[3]]

bars_lr = ax2.bar(categories_lr, probabilities_lr, color=colors_lr, width=0.6)

for bar_lr, value_lr in zip(bars_lr, probabilities_lr):
    height_lr = bar_lr.get_height()
    ax2.text(bar_lr.get_x() + bar_lr.get_width()/2., height_lr + 0.02,
            f'{value_lr*100:.1f}%', ha='center', va='bottom', color=TEXT_PRIMARY, 
            fontsize=10, fontweight='bold')

ax2.set_ylabel('Probability', fontsize=11, color=TEXT_PRIMARY, fontweight='bold')
ax2.set_title('Scenario 2: Low-Risk Customer\\nChurn Probability', fontsize=12, 
             color=TEXT_PRIMARY, fontweight='bold', pad=15)
ax2.set_ylim(0, 1.1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color(TEXT_SECONDARY)
ax2.spines['bottom'].set_color(TEXT_SECONDARY)
ax2.tick_params(colors=TEXT_PRIMARY, labelsize=9)

# Risk level comparison
ax3 = plt.subplot(2, 3, 3)
ax3.set_facecolor(BG_COLOR)

scenarios_comp = ['High-Risk\\nScenario', 'Low-Risk\\nScenario']
risk_probs = [result_hr['churn_probability'], result_lr['churn_probability']]
risk_colors = [ZERVE_COLORS[3], ZERVE_COLORS[2]]

bars_comp = ax3.barh(scenarios_comp, risk_probs, color=risk_colors)

for bar_comp, value_comp in zip(bars_comp, risk_probs):
    width_comp = bar_comp.get_width()
    ax3.text(width_comp + 0.02, bar_comp.get_y() + bar_comp.get_height()/2,
            f'{value_comp*100:.1f}%', va='center', color=TEXT_PRIMARY, 
            fontsize=10, fontweight='bold')

ax3.set_xlabel('Churn Probability', fontsize=11, color=TEXT_PRIMARY, fontweight='bold')
ax3.set_title('Risk Level Comparison', fontsize=12, color=TEXT_PRIMARY, 
             fontweight='bold', pad=15)
ax3.set_xlim(0, 1.1)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color(TEXT_SECONDARY)
ax3.spines['bottom'].set_color(TEXT_SECONDARY)
ax3.tick_params(colors=TEXT_PRIMARY, labelsize=9)

# Model metrics summary from training
ax4 = plt.subplot(2, 3, 4)
ax4.set_facecolor(BG_COLOR)

model_metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
model_metrics_values = [model_accuracy, model_precision, model_recall, model_f1, model_roc_auc]

bars_metrics = ax4.barh(model_metrics_names, model_metrics_values, color=ZERVE_COLORS[0])

for bar_m, value_m in zip(bars_metrics, model_metrics_values):
    width_m = bar_m.get_width()
    ax4.text(width_m + 0.02, bar_m.get_y() + bar_m.get_height()/2,
            f'{value_m:.3f}', va='center', color=TEXT_PRIMARY, 
            fontsize=9, fontweight='bold')

ax4.set_xlabel('Score', fontsize=11, color=TEXT_PRIMARY, fontweight='bold')
ax4.set_title('Model Performance Metrics', fontsize=12, color=TEXT_PRIMARY, 
             fontweight='bold', pad=15)
ax4.set_xlim(0, 1.1)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_color(TEXT_SECONDARY)
ax4.spines['bottom'].set_color(TEXT_SECONDARY)
ax4.tick_params(colors=TEXT_PRIMARY, labelsize=9)

# Customer characteristics comparison
ax5 = plt.subplot(2, 3, 5)
ax5.set_facecolor(BG_COLOR)

char_names = ['Tenure\\n(months)', 'Monthly\\nCharges ($)', 'Senior\\nCitizen']
hr_chars = [high_risk_customer_web['tenure'], high_risk_customer_web['MonthlyCharges'], 
            high_risk_customer_web['SeniorCitizen']]
lr_chars = [low_risk_customer_web['tenure'], low_risk_customer_web['MonthlyCharges'], 
            low_risk_customer_web['SeniorCitizen']]

# Normalize values for comparison
hr_chars_norm = [hr_chars[0]/72, hr_chars[1]/120, hr_chars[2]]
lr_chars_norm = [lr_chars[0]/72, lr_chars[1]/120, lr_chars[2]]

x_pos = range(len(char_names))
width_compare = 0.35

bars_hr_char = ax5.bar([p - width_compare/2 for p in x_pos], hr_chars_norm, 
                       width_compare, label='High-Risk', color=ZERVE_COLORS[3])
bars_lr_char = ax5.bar([p + width_compare/2 for p in x_pos], lr_chars_norm, 
                       width_compare, label='Low-Risk', color=ZERVE_COLORS[2])

ax5.set_ylabel('Normalized Value', fontsize=11, color=TEXT_PRIMARY, fontweight='bold')
ax5.set_title('Customer Characteristics Comparison', fontsize=12, color=TEXT_PRIMARY, 
             fontweight='bold', pad=15)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(char_names, fontsize=9)
ax5.legend(loc='upper right', framealpha=0.9, facecolor=BG_COLOR, 
          edgecolor=TEXT_SECONDARY, labelcolor=TEXT_PRIMARY, fontsize=9)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_color(TEXT_SECONDARY)
ax5.spines['bottom'].set_color(TEXT_SECONDARY)
ax5.tick_params(colors=TEXT_PRIMARY, labelsize=9)

# Confidence comparison
ax6 = plt.subplot(2, 3, 6)
ax6.set_facecolor(BG_COLOR)

scenarios_conf = ['High-Risk\\nScenario', 'Low-Risk\\nScenario']
confidences = [result_hr['confidence'], result_lr['confidence']]

bars_conf = ax6.bar(scenarios_conf, confidences, color=[ZERVE_COLORS[3], ZERVE_COLORS[2]], 
                   width=0.5)

for bar_c, value_c in zip(bars_conf, confidences):
    height_c = bar_c.get_height()
    ax6.text(bar_c.get_x() + bar_c.get_width()/2., height_c + 0.02,
            f'{value_c:.3f}', ha='center', va='bottom', color=TEXT_PRIMARY, 
            fontsize=10, fontweight='bold')

ax6.set_ylabel('Confidence', fontsize=11, color=TEXT_PRIMARY, fontweight='bold')
ax6.set_title('Model Confidence Comparison', fontsize=12, color=TEXT_PRIMARY, 
             fontweight='bold', pad=15)
ax6.set_ylim(0, 1.1)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_color(TEXT_SECONDARY)
ax6.spines['bottom'].set_color(TEXT_SECONDARY)
ax6.tick_params(colors=TEXT_PRIMARY, labelsize=9)

plt.tight_layout()
comprehensive_results = fig_combined
print("\\n✅ Comprehensive visualization created")

# Results Summary Table
print("\\n" + "=" * 80)
print("📋 COMPLETE RESULTS TABLE")
print("=" * 80)

results_summary_data = {
    'Metric': [
        'Churn Prediction',
        'Churn Probability',
        'Risk Level',
        'Confidence',
        'Tenure',
        'Monthly Charges',
        'Contract Type',
        'Tech Support'
    ],
    'High-Risk Customer': [
        result_hr['churn_prediction'],
        result_hr['churn_probability_pct'],
        result_hr['risk_level'],
        f"{result_hr['confidence']}",
        f"{high_risk_customer_web['tenure']} months",
        f"${high_risk_customer_web['MonthlyCharges']:.2f}",
        high_risk_customer_web['Contract'],
        high_risk_customer_web['TechSupport']
    ],
    'Low-Risk Customer': [
        result_lr['churn_prediction'],
        result_lr['churn_probability_pct'],
        result_lr['risk_level'],
        f"{result_lr['confidence']}",
        f"{low_risk_customer_web['tenure']} months",
        f"${low_risk_customer_web['MonthlyCharges']:.2f}",
        low_risk_customer_web['Contract'],
        low_risk_customer_web['TechSupport']
    ]
}

results_summary_df = pd.DataFrame(results_summary_data)
print("\\n" + results_summary_df.to_string(index=False))

print("\\n" + "=" * 80)
print("✅ WEB APPLICATION COMPLETE")
print("=" * 80)
print("\\n📝 Summary:")
print("  ✓ Interactive prediction system integrated with ML model API")
print("  ✓ Comprehensive result visualization including:")
print("    • Individual scenario predictions with probability distributions")
print("    • Risk level comparisons across scenarios")
print("    • Model performance metrics from training")
print("    • Customer characteristic comparisons")
print("    • Confidence level analysis")
print("  ✓ Detailed results table with key metrics and customer attributes")
print("  ✓ Actionable recommendations for each risk level")
print("\\n💡 To use with custom data:")
print("  1. Create customer data dictionary with all required fields")
print("  2. Call: result = prediction_service.predict(customer_data)")
print("  3. Access results: result['churn_prediction'], result['churn_probability_pct'], etc.")
print("=" * 80)