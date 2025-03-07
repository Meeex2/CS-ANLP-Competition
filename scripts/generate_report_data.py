import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Check if the 'report' directory exists, if not create it
report_dir = "report"
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

# Load the classification report CSV
df = pd.read_csv("output/classification_report_with_names.csv")

# Remove the 'support' column as it's not necessary for visualization
df = df.drop(columns=["support"])

# Round numeric columns to 2 decimal places
df = df.round(2)

# Sort the data by F1-score in descending order
df = df.sort_values(by="f1-score", ascending=False)

# Number of labels per plot
labels_per_plot = 15

# Calculate number of subplots needed
num_plots = int(np.ceil(len(df) / labels_per_plot))

# Define grid dimensions (we'll use 5 columns by default, adjust based on num_plots)
num_columns = 5
num_rows = int(np.ceil(num_plots / num_columns))

# Create subplots for F1-scores
fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 2))
axes = axes.flatten()

for i in range(num_plots):
    start = i * labels_per_plot
    end = start + labels_per_plot
    subset = df.iloc[start:end]

    sns.barplot(x=subset["Label"], y=subset["f1-score"], ax=axes[i], palette="Set3")
    axes[i].set_title(f"F1-Score Distribution\n(Labels {start + 1} to {end})")
    axes[i].set_xlabel("Label")
    axes[i].set_ylabel("F1-Score")
    axes[i].tick_params(axis="x", rotation=90)

for j in range(num_plots, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
f1_score_grid_path = os.path.join(report_dir, "f1_score_grid.png")
plt.savefig(f1_score_grid_path)
plt.close()

# Create subplots for recall
fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 2))
axes = axes.flatten()

for i in range(num_plots):
    start = i * labels_per_plot
    end = start + labels_per_plot
    subset = df.iloc[start:end]

    sns.barplot(x=subset["Label"], y=subset["recall"], ax=axes[i], palette="Set2")
    axes[i].set_title(f"Recall Distribution\n(Labels {start + 1} to {end})")
    axes[i].set_xlabel("Label")
    axes[i].set_ylabel("Recall")
    axes[i].tick_params(axis="x", rotation=90)

for j in range(num_plots, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
recall_grid_path = os.path.join(report_dir, "recall_grid.png")
plt.savefig(recall_grid_path)
plt.close()

# Create subplots for precision
fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 2))
axes = axes.flatten()

for i in range(num_plots):
    start = i * labels_per_plot
    end = start + labels_per_plot
    subset = df.iloc[start:end]

    sns.barplot(x=subset["Label"], y=subset["precision"], ax=axes[i], palette="Set1")
    axes[i].set_title(f"Precision Distribution\n(Labels {start + 1} to {end})")
    axes[i].set_xlabel("Label")
    axes[i].set_ylabel("Precision")
    axes[i].tick_params(axis="x", rotation=90)

for j in range(num_plots, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
precision_grid_path = os.path.join(report_dir, "precision_grid.png")
plt.savefig(precision_grid_path)
plt.close()

# Save the LaTeX table for the full classification report (without 'support' column)
latex_table_full = df.to_latex(
    index=False,
    float_format="%.2f",
    caption="Full Classification Report",
    label="tab:full_classification_report",
)

# Define output path for the LaTeX table
output_path_full = os.path.join(report_dir, "full_classification_report.tex")

# Save the LaTeX table to file
with open(output_path_full, "w") as f_full:
    f_full.write(latex_table_full)

# Output success messages
print(f"F1-Score distribution grid plot saved to {f1_score_grid_path}")
print(f"Recall distribution grid plot saved to {recall_grid_path}")
print(f"Precision distribution grid plot saved to {precision_grid_path}")
print(f"LaTeX table for Full Classification Report saved to {output_path_full}")
