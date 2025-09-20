import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

locality = "local_1"
tech = "neo"
tech = "ms"
grouping = "grouped"
grouping = "each"
file_name = f"{tech}_{grouping}"
path = f"data/{locality}/{file_name}.json"

with open(path, "r", encoding="utf-8") as f:
    stats = json.load(f)

# Flatten JSON into a DataFrame
flat = []
for group_idx, group in enumerate(stats):
    for metric, stats_dict in group.items():
        flat.append({
            "q": f"q {group_idx+1}",
            "metric": metric,
            **stats_dict
        })

df = pd.DataFrame(flat)

# Get unique groups/qs
groups = df["q"].unique()
n_groups = len(groups)

# Loop over each metric
metrics = df["metric"].unique()
# Loop over each metric
for metric in metrics:
    df_metric = df[df["metric"] == metric]
    
    n_groups = len(df_metric["q"].unique())
    x_positions = np.arange(n_groups)
    if n_groups > 20:
        width = 50 / n_groups  # bar width scaling
    else:
        width = 8 / n_groups  # bar width scaling
    
    overall_min = df_metric["min"].min()
    overall_max = df_metric["max"].max()
    
    if abs(overall_min - overall_max) < 1:
        padding = overall_min * 0.01 if overall_min != 0 else 1
        overall_min -= padding
        overall_max += padding
    
    # Compute overall min/max for y-axis scaling
    y_min = max(0, overall_min - 0.05*(overall_max - overall_min))
    y_max = overall_max + 0.05*(overall_max - overall_min)  # small upper padding

    # Handle zero variance
    if abs(y_min - y_max) < 1:
        padding = y_min * 0.01 if y_min != 0 else 1
        y_min -= padding
        y_max += padding
    
    fig_width = max(10, n_groups * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    whisker_cap_width = width / 2  # width of horizontal caps
    
    for i, group in enumerate(df_metric["q"].unique()):
        row = df_metric[df_metric["q"] == group].iloc[0]
        
        q25 = row["q25"]
        q75 = row["q75"]
        median = row["median"]
        mean = row["mean"]
        min_val = row["min"]
        max_val = row["max"]
        
        # Draw bar from q25 to q75
        ax.bar(i, q75-q25, bottom=q25, width=width, color="skyblue", edgecolor="black")
        
        # Draw whiskers (min/max)
        ax.vlines(i, min_val, max_val, color="black", linewidth=1.5)
        
        # Draw whisker caps
        ax.hlines(min_val, i - whisker_cap_width/2, i + whisker_cap_width/2, color="black", linewidth=1.5)
        ax.hlines(max_val, i - whisker_cap_width/2, i + whisker_cap_width/2, color="black", linewidth=1.5)
        
        # Draw median line
        ax.hlines(median, i - width/2, i + width/2, color="red", linewidth=2)
        
        # Draw median line
        ax.hlines(mean, i - width/2, i + width/2, color="lime", linewidth=2)

    # Center the bars
    padding = 0.5
    ax.set_xlim(-padding, n_groups - 1 + padding)
    
    # X-axis labels
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(df_metric["q"].unique(), rotation=45, ha="right")
    
    ax.set_ylabel("Value")
    ax.set_title(f"Metric: {metric}")
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    if not os.path.exists(f"plots/{locality}/{tech}_{grouping}"):
        os.makedirs(f"plots/{locality}/{tech}_{grouping}")
    plt.savefig(f"plots/l{locality}/{tech}_{grouping}/{metric}.pdf", format="pdf")
    plt.savefig(f"plots/{locality}/{tech}_{grouping}/{metric}.svg", format="svg")
    plt.close()
