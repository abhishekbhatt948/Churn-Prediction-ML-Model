import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set Zerve design system colors
BG_COLOR = '#1D1D20'
TEXT_PRIMARY = '#fbfbff'
TEXT_SECONDARY = '#909094'
ZERVE_COLORS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#1F77B4', '#9467BD', '#8C564B']
HIGHLIGHT = '#ffd400'

# Churn Distribution
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor(BG_COLOR)
ax1.set_facecolor(BG_COLOR)

churn_counts = df_clean['Churn'].value_counts()
bars = ax1.bar(churn_counts.index, churn_counts.values, color=[ZERVE_COLORS[2], ZERVE_COLORS[3]], width=0.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/len(df_clean)*100:.1f}%)',
            ha='center', va='bottom', color=TEXT_PRIMARY, fontsize=11, fontweight='bold')

ax1.set_xlabel('Churn Status', fontsize=12, color=TEXT_PRIMARY, fontweight='bold')
ax1.set_ylabel('Number of Customers', fontsize=12, color=TEXT_PRIMARY, fontweight='bold')
ax1.set_title('Customer Churn Distribution', fontsize=14, color=TEXT_PRIMARY, fontweight='bold', pad=20)
ax1.tick_params(colors=TEXT_PRIMARY, labelsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color(TEXT_SECONDARY)
ax1.spines['bottom'].set_color(TEXT_SECONDARY)
plt.tight_layout()
churn_dist_chart = fig1
print("✓ Churn distribution chart created")

# Churn Rate by Key Categorical Features
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.patch.set_facecolor(BG_COLOR)

key_features = ['Contract', 'InternetService', 'TechSupport', 'PaymentMethod']
titles = ['Churn Rate by Contract Type', 'Churn Rate by Internet Service', 
          'Churn Rate by Tech Support', 'Churn Rate by Payment Method']

for idx, (feature, title) in enumerate(zip(key_features, titles)):
    ax = axes[idx // 2, idx % 2]
    ax.set_facecolor(BG_COLOR)
    
    # Calculate churn rate for each category
    churn_rate = df_clean.groupby(feature)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
    
    bars = ax.barh(range(len(churn_rate)), churn_rate.values, color=ZERVE_COLORS[0])
    ax.set_yticks(range(len(churn_rate)))
    ax.set_yticklabels(churn_rate.index, fontsize=9)
    
    # Add percentage labels
    for i, (val, label) in enumerate(zip(churn_rate.values, churn_rate.index)):
        ax.text(val + 1, i, f'{val:.1f}%', va='center', color=TEXT_PRIMARY, fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Churn Rate (%)', fontsize=10, color=TEXT_PRIMARY, fontweight='bold')
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_PRIMARY, labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(TEXT_SECONDARY)
    ax.spines['bottom'].set_color(TEXT_SECONDARY)

plt.tight_layout()
churn_by_features_chart = fig2
print("✓ Churn rate by key features charts created")

# Churn Rate by Tenure Groups
fig3, ax3 = plt.subplots(figsize=(12, 6))
fig3.patch.set_facecolor(BG_COLOR)
ax3.set_facecolor(BG_COLOR)

# Create tenure groups
tenure_bins = [0, 12, 24, 36, 48, 72]
tenure_labels = ['0-12 months', '13-24 months', '25-36 months', '37-48 months', '49+ months']
df_clean['TenureGroup'] = pd.cut(df_clean['tenure'], bins=tenure_bins, labels=tenure_labels, include_lowest=True)

tenure_churn = df_clean.groupby('TenureGroup')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)

bars = ax3.bar(range(len(tenure_churn)), tenure_churn.values, color=ZERVE_COLORS[1], width=0.6)
ax3.set_xticks(range(len(tenure_churn)))
ax3.set_xticklabels(tenure_churn.index, rotation=0, ha='center')

# Add percentage labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', color=TEXT_PRIMARY, fontsize=10, fontweight='bold')

ax3.set_xlabel('Tenure Period', fontsize=12, color=TEXT_PRIMARY, fontweight='bold')
ax3.set_ylabel('Churn Rate (%)', fontsize=12, color=TEXT_PRIMARY, fontweight='bold')
ax3.set_title('Churn Rate Decreases with Customer Tenure', fontsize=14, color=TEXT_PRIMARY, fontweight='bold', pad=20)
ax3.tick_params(colors=TEXT_PRIMARY, labelsize=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color(TEXT_SECONDARY)
ax3.spines['bottom'].set_color(TEXT_SECONDARY)
plt.tight_layout()
tenure_churn_chart = fig3
print("✓ Churn rate by tenure chart created")

print("\n" + "=" * 70)
print("KEY INSIGHTS - CHURN DRIVERS")
print("=" * 70)
print(f"\n1. Overall Churn Rate: {(df_clean['Churn'] == 'Yes').mean() * 100:.1f}%")
print(f"\n2. Contract Type Impact:")
for contract in df_clean['Contract'].unique():
    rate = (df_clean[df_clean['Contract'] == contract]['Churn'] == 'Yes').mean() * 100
    print(f"   - {contract}: {rate:.1f}% churn rate")
    
print(f"\n3. Tenure Effect:")
print(f"   - Early customers (0-12 months): {tenure_churn.iloc[0]:.1f}% churn")
print(f"   - Long-term customers (49+ months): {tenure_churn.iloc[-1]:.1f}% churn")
print(f"   - Risk reduction: {tenure_churn.iloc[0] - tenure_churn.iloc[-1]:.1f}% lower churn for loyal customers")

print(f"\n4. Tech Support Importance:")
tech_with = (df_clean[df_clean['TechSupport'] == 'Yes']['Churn'] == 'Yes').mean() * 100
tech_without = (df_clean[df_clean['TechSupport'] == 'No']['Churn'] == 'Yes').mean() * 100
print(f"   - With tech support: {tech_with:.1f}% churn")
print(f"   - Without tech support: {tech_without:.1f}% churn")
