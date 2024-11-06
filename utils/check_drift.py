import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import os
from datetime import datetime



new_data = pd.read_csv("temp/simple_milling_data.csv")

# 1. Compare summary statistics
#print("Original Data Summary Statistics:")
#print(data.describe())

print("\nNew Data Summary Statistics:")
print(new_data.describe())

# 2. Visualize data distribution comparison with KDE plots for drift detection
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 16))
axes = axes.flatten()

for idx, col in enumerate(old_data.columns):
    sns.kdeplot(old_data[col], ax=axes[idx], label='Original Data', color='blue')
    sns.kdeplot(new_data[col], ax=axes[idx], label='New Data', color='orange')
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].legend()

plt.tight_layout()

# Create directory if it doesn't exist
save_path = "metrics/drift"
os.makedirs(save_path, exist_ok=True)
plot_path = os.path.join(save_path, f'data_distribution{datetime.now().date()}.png')

# Save the plot
plt.savefig(plot_path)


# 3. Statistical Test for Drift: Calculate KL Divergence between original and new data for each column
kl_divergence = {}
for col in data.columns:
    # Use histogram bins for KL divergence calculation
    hist_original, bin_edges = np.histogram(data[col], bins=20, density=True)
    hist_new, _ = np.histogram(new_data[col], bins=bin_edges, density=True)
    
    # Avoid division by zero by adding a small value to zero bins
    kl_divergence[col] = entropy(hist_original + 1e-10, hist_new + 1e-10)

print("\nKL Divergence for each feature:")
for col, kl_val in kl_divergence.items():
    print(f"{col}: {kl_val:.4f}")

# Interpretation: Higher KL Divergence indicates significant drift in that feature
