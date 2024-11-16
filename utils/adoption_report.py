import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from the CSV files
file1 = "metrics/old_model_performance.csv"  # Replace with your first CSV file path
file2 = "metrics/new_model_performance.csv"  # Replace with your second CSV file path

df1 = pd.read_csv(file1)  # Old values
df2 = pd.read_csv(file2)  # New values

# Ensure both DataFrames have the same columns
if not all(df1.columns == df2.columns):
    raise ValueError("The two files must have identical columns for comparison.")

# Average the data for comparison (if needed, adjust logic as per data structure)
df1_avg = df1.mean()
df2_avg = df2.mean()

# Plot settings
x = np.arange(len(df1_avg))  # Positions for the bars
width = 0.35  # Width of the bars

plt.figure(figsize=(12, 6))

# Create bars for old values
plt.bar(x - width / 2, df1_avg, width, label='Old Values', color='blue')

# Create bars for new values
plt.bar(x + width / 2, df2_avg, width, label='New Values', color='orange')

# Add labels, title, and legend
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Comparison of Old and New Values')
plt.xticks(ticks=x, labels=df1.columns, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()