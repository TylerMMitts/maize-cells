# How strongly does stele area track root radius?
#
# Establishes that one can stand in for the other, which is what justifies
# predicting stele area rather than measuring it.

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from code.config import MASTER_SUMMARY_PATH, RESULTS_FOLDER
warnings.filterwarnings('ignore')

# Define file path
file_path = Path(MASTER_SUMMARY_PATH)

# Load the data
df = pd.read_csv(file_path)

# Check for missing values in the two variables of interest
missing_radius = df['root_radius_um'].isna().sum()
missing_stele = df['stele_area_um2'].isna().sum()

print(f"\nMissing values:")
print(f"  root_radius_um: {missing_radius} missing")
print(f"  stele_area_um2: {missing_stele} missing")

# Remove rows with missing values for these variables
df_clean = df.dropna(subset=['root_radius_um', 'stele_area_um2'])

print(f"\nObservations after removing missing values: {len(df_clean)} rows")

print("\nRoot Radius (µm):")
print(df_clean['root_radius_um'].describe())

print("\nStele Area (µm²):")
print(df_clean['stele_area_um2'].describe())

# Extract the two variables
radius = df_clean['root_radius_um'].values
stele = df_clean['stele_area_um2'].values

# 3a. Pearson Correlation (linear relationship)
pearson_r, pearson_p = pearsonr(radius, stele)

print(f"\nPEARSON CORRELATION (Linear):")
print(f"  Correlation coefficient (r): {pearson_r:.4f}")
print(f"  P-value: {pearson_p:.4e}")
print(f"  R-squared (r²): {pearson_r**2:.4f}")
print(f"  Interpretation: {pearson_r**2*100:.1f}% of the variance in stele area")
print(f"                  can be explained by root radius")

# 3b. Spearman Correlation (monotonic relationship - more robust to outliers)
spearman_r, spearman_p = spearmanr(radius, stele)

print(f"\nSPEARMAN CORRELATION (Rank-based, robust):")
print(f"  Correlation coefficient (ρ): {spearman_r:.4f}")
print(f"  P-value: {spearman_p:.4e}")

# 3c. Interpretation based on correlation strength
def interpret_correlation(r):
    abs_r = abs(r)
    if abs_r < 0.3:
        return "Weak"
    elif abs_r < 0.5:
        return "Moderate"
    elif abs_r < 0.7:
        return "Strong"
    else:
        return "Very Strong"

pearson_strength = interpret_correlation(pearson_r)
print(f"\nInterpretation:")
print(f"  Pearson correlation is {pearson_strength} (r = {pearson_r:.3f})")


# Create a figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Root Radius vs Stele Area: Correlation Analysis', fontsize=16, fontweight='bold')

# 4a. Scatter plot with regression line
ax1 = axes[0, 0]
ax1.scatter(radius, stele, alpha=0.5, s=20, label='Data points')

# Add regression line
z = np.polyfit(radius, stele, 1)
p = np.poly1d(z)
ax1.plot(radius, p(radius), "r-", linewidth=2, label=f'Regression: r² = {pearson_r**2:.3f}')

ax1.set_xlabel('Root Radius (µm)', fontsize=12)
ax1.set_ylabel('Stele Area (µm²)', fontsize=12)
ax1.set_title('Scatter Plot with Linear Regression', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 4b. Hexbin plot (better for dense data)
ax2 = axes[0, 1]
hb = ax2.hexbin(radius, stele, gridsize=30, cmap='viridis', mincnt=1)
ax2.set_xlabel('Root Radius (µm)', fontsize=12)
ax2.set_ylabel('Stele Area (µm²)', fontsize=12)
ax2.set_title('Hexbin Plot (density visualization)', fontsize=13)
plt.colorbar(hb, ax=ax2, label='Count')

# 4c. Residuals plot
ax3 = axes[1, 0]
# Calculate residuals
residuals = stele - p(radius)
ax3.scatter(p(radius), residuals, alpha=0.5, s=20)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Stele Area (µm²)', fontsize=12)
ax3.set_ylabel('Residuals (µm²)', fontsize=12)
ax3.set_title('Residuals Plot (check for patterns)', fontsize=13)
ax3.grid(True, alpha=0.3)

# 4d. Distribution comparison
ax4 = axes[1, 1]
# Normalize both variables for comparison
radius_norm = (radius - radius.mean()) / radius.std()
stele_norm = (stele - stele.mean()) / stele.std()

ax4.hist(radius_norm, bins=30, alpha=0.5, label='Root Radius (normalized)', color='blue')
ax4.hist(stele_norm, bins=30, alpha=0.5, label='Stele Area (normalized)', color='green')
ax4.set_xlabel('Z-score (normalized values)', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Distribution Comparison (normalized)', fontsize=13)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
output_dir = Path(RESULTS_FOLDER / 'radius_stele_correlation_analysis')
output_dir.mkdir(exist_ok=True)
fig_path = output_dir / 'root_radius_vs_stele_correlation.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {fig_path}")


# 5a. By treatment
if 'treatment' in df_clean.columns:
    print("\nCorrelation by Treatment:")
    treatments = df_clean['treatment'].unique()
    for treatment in treatments:
        subset = df_clean[df_clean['treatment'] == treatment]
        if len(subset) > 5:  # Only calculate if enough data
            r, p = pearsonr(subset['root_radius_um'], subset['stele_area_um2'])
            print(f"  {treatment} (n={len(subset)}): r = {r:.4f}, p = {p:.4e}")

# 5b. By species/population
if 'species' in df_clean.columns:
    print("\nCorrelation by Species:")
    species_list = df_clean['species'].unique()
    for sp in species_list:
        subset = df_clean[df_clean['species'] == sp]
        if len(subset) > 5:
            r, p = pearsonr(subset['root_radius_um'], subset['stele_area_um2'])
            print(f"  {sp} (n={len(subset)}): r = {r:.4f}, p = {p:.4e}")


# Calculate Variance Inflation Factor (VIF) for root radius
# (Simplified: VIF for a single predictor is 1/(1-r²))
vif = 1 / (1 - pearson_r**2)


print(f"""
Key Findings:
-------------
1. Pearson correlation: r = {pearson_r:.4f} (r² = {pearson_r**2:.4f})
2. Spearman correlation: ρ = {spearman_r:.4f}
3. Sample size: n = {len(df_clean)}
4. Correlation strength: {pearson_strength}
""")

# Create a results dataframe
results_df = pd.DataFrame({
    'test': ['Pearson', 'Spearman'],
    'coefficient': [pearson_r, spearman_r],
    'p_value': [pearson_p, spearman_p],
    'n': [len(df_clean), len(df_clean)],
    'interpretation': [pearson_strength, pearson_strength]
})

results_path = output_dir / 'correlation_results.csv'
results_df.to_csv(results_path, index=False)
print(f"\nResults exported to: {results_path}")

# Also save the clean data for future use
clean_data_path = output_dir / 'master_summary_clean.csv'
df_clean.to_csv(clean_data_path, index=False)
print(f"Clean data saved to: {clean_data_path}")

# Show the plot
plt.show()