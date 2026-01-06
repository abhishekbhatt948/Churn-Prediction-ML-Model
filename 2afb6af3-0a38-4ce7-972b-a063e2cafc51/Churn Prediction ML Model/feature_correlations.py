import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Zerve design system colors
BG_COLOR = '#1D1D20'
TEXT_PRIMARY = '#fbfbff'
TEXT_SECONDARY = '#909094'
ZERVE_COLORS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']

# Prepare numerical features for correlation analysis
# Convert binary categorical to numeric
df_numeric = df_clean.copy()

# Binary encoding for key categorical features
binary_mappings = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'Churn': {'Yes': 1, 'No': 0}
}

for col, mapping in binary_mappings.items():
    df_numeric[col] = df_numeric[col].map(mapping)

# Multi-category encoding for internet service
df_numeric['HasInternetService'] = (df_numeric['InternetService'] != 'No').astype(int)
df_numeric['IsFiberOptic'] = (df_numeric['InternetService'] == 'Fiber optic').astype(int)

# Contract type encoding
df_numeric['IsMonthToMonth'] = (df_numeric['Contract'] == 'Month-to-month').astype(int)
df_numeric['HasLongContract'] = ((df_numeric['Contract'] == 'One year') | (df_numeric['Contract'] == 'Two year')).astype(int)

# Tech support encoding
df_numeric['HasTechSupport'] = (df_numeric['TechSupport'] == 'Yes').astype(int)
df_numeric['HasOnlineSecurity'] = (df_numeric['OnlineSecurity'] == 'Yes').astype(int)

# Select numerical and encoded features for correlation
correlation_features = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                       'PhoneService', 'HasInternetService', 'IsFiberOptic',
                       'HasTechSupport', 'HasOnlineSecurity', 'IsMonthToMonth',
                       'HasLongContract', 'PaperlessBilling', 
                       'MonthlyCharges', 'TotalCharges', 'Churn']

corr_data = df_numeric[correlation_features]

# Calculate correlation matrix
corr_matrix = corr_data.corr()

# Correlation Heatmap
fig1, ax1 = plt.subplots(figsize=(14, 11))
fig1.patch.set_facecolor(BG_COLOR)
ax1.set_facecolor(BG_COLOR)

# Create heatmap with custom colormap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            ax=ax1, vmin=-0.5, vmax=0.5)

ax1.set_title('Feature Correlation Matrix with Churn', fontsize=14, color=TEXT_PRIMARY, fontweight='bold', pad=20)
ax1.tick_params(colors=TEXT_PRIMARY, labelsize=9)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', color=TEXT_PRIMARY)
plt.setp(ax1.get_yticklabels(), rotation=0, color=TEXT_PRIMARY)

# Customize colorbar
cbar = ax1.collections[0].colorbar
cbar.ax.tick_params(colors=TEXT_PRIMARY, labelsize=9)
cbar.outline.set_edgecolor(TEXT_SECONDARY)

plt.tight_layout()
correlation_heatmap = fig1
print("✓ Correlation heatmap created")

# Top correlations with churn
fig2, ax2 = plt.subplots(figsize=(11, 7))
fig2.patch.set_facecolor(BG_COLOR)
ax2.set_facecolor(BG_COLOR)

churn_correlations = corr_matrix['Churn'].drop('Churn').sort_values(ascending=True)
colors = [ZERVE_COLORS[3] if x > 0 else ZERVE_COLORS[2] for x in churn_correlations.values]

bars = ax2.barh(range(len(churn_correlations)), churn_correlations.values, color=colors)
ax2.set_yticks(range(len(churn_correlations)))
ax2.set_yticklabels(churn_correlations.index, fontsize=9)

# Add correlation values
for i, val in enumerate(churn_correlations.values):
    x_pos = val + (0.01 if val > 0 else -0.01)
    ha = 'left' if val > 0 else 'right'
    ax2.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, color=TEXT_PRIMARY, fontsize=9, fontweight='bold')

ax2.axvline(x=0, color=TEXT_SECONDARY, linestyle='-', linewidth=1)
ax2.set_xlabel('Correlation Coefficient', fontsize=11, color=TEXT_PRIMARY, fontweight='bold')
ax2.set_title('Feature Correlations with Churn', fontsize=13, color=TEXT_PRIMARY, fontweight='bold', pad=20)
ax2.tick_params(colors=TEXT_PRIMARY, labelsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color(TEXT_SECONDARY)
ax2.spines['bottom'].set_color(TEXT_SECONDARY)
plt.tight_layout()
churn_correlation_chart = fig2
print("✓ Churn correlation chart created")

# Numerical feature distributions by churn status
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.patch.set_facecolor(BG_COLOR)

num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
feature_labels = ['Tenure (months)', 'Monthly Charges ($)', 'Total Charges ($)', 'Senior Citizen']

for idx, (feat, feat_label) in enumerate(zip(num_features, feature_labels)):
    ax = axes3[idx // 2, idx % 2]
    ax.set_facecolor(BG_COLOR)
    
    churned = df_numeric[df_numeric['Churn'] == 1][feat]
    retained = df_numeric[df_numeric['Churn'] == 0][feat]
    
    if feat == 'SeniorCitizen':
        # Bar chart for binary feature
        churn_senior = df_numeric.groupby([feat, 'Churn']).size().unstack(fill_value=0)
        churn_pct = churn_senior.div(churn_senior.sum(axis=1), axis=0) * 100
        
        x = np.arange(2)
        width = 0.35
        bars1 = ax.bar(x - width/2, churn_pct[0], width, label='Retained', color=ZERVE_COLORS[2])
        bars2 = ax.bar(x + width/2, churn_pct[1], width, label='Churned', color=ZERVE_COLORS[3])
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Not Senior', 'Senior'])
        ax.set_ylabel('Percentage (%)', fontsize=10, color=TEXT_PRIMARY, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9, facecolor=BG_COLOR, edgecolor=TEXT_SECONDARY, labelcolor=TEXT_PRIMARY)
    else:
        # Histogram for continuous features
        bins = 30 if feat != 'tenure' else 20
        ax.hist(retained, bins=bins, alpha=0.7, label='Retained', color=ZERVE_COLORS[2], edgecolor=BG_COLOR)
        ax.hist(churned, bins=bins, alpha=0.7, label='Churned', color=ZERVE_COLORS[3], edgecolor=BG_COLOR)
        ax.set_ylabel('Frequency', fontsize=10, color=TEXT_PRIMARY, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9, facecolor=BG_COLOR, edgecolor=TEXT_SECONDARY, labelcolor=TEXT_PRIMARY)
    
    ax.set_xlabel(feat_label, fontsize=10, color=TEXT_PRIMARY, fontweight='bold')
    ax.set_title(f'{feat_label} Distribution by Churn Status', fontsize=11, color=TEXT_PRIMARY, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_PRIMARY, labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(TEXT_SECONDARY)
    ax.spines['bottom'].set_color(TEXT_SECONDARY)

plt.tight_layout()
distribution_by_churn = fig3
print("✓ Feature distribution charts created")

print("\n" + "=" * 70)
print("CORRELATION INSIGHTS")
print("=" * 70)
print("\nTop Positive Correlations with Churn (increase churn risk):")
top_positive = churn_correlations.tail(5)
for feat, corr in top_positive.items():
    print(f"  • {feat}: {corr:.3f}")

print("\nTop Negative Correlations with Churn (decrease churn risk):")
top_negative = churn_correlations.head(5)
for feat, corr in top_negative.items():
    print(f"  • {feat}: {corr:.3f}")

print("\n" + "=" * 70)
