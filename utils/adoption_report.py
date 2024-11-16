import pandas as pd
import matplotlib.pyplot as plt

# Load data from the CSV files
file1 = "metrics/old_model_performance.csv"  # Replace with your first CSV file path
file2 = "metrics/new_model_performance.csv"  # Replace with your second CSV file path

# Read CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Define a fixed color cycle
colors = ['red', 'green', 'blue', 'black']

# Plot settings
plt.figure(figsize=(10, 6))

# Plot data from the first file with dotted lines
for i, column in enumerate(df1.columns):
    plt.plot(df1.index, df1[column], linestyle=':', color=colors[i % len(colors)], label=f'{column} (dotted)')

# Plot data from the second file with solid lines
for i, column in enumerate(df2.columns):
    plt.plot(df2.index, df2[column], linestyle='-', color=colors[i % len(colors)], label=f'{column} (solid)')

# Add labels, title, legend, and grid
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Comparison of Two CSV Files')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
