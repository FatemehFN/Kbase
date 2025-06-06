import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import xnetwork as xn

# Directory where Leiden clustering results are stored
base_dir = "/Users/fsfatemi/local_AD/KNN_NetworksWithCommunities"
resolutions = [0.001, 0.005, 0.01, 0.1, 1.0, 0.5, 5.0, 10.0, 20.0, 50, 100]
ks = list(range(4, 16))

# Prepare a DataFrame to collect results
results = []

# Read Leiden results from each knn_x.xnet folder
for k in ks:
    file = f"{base_dir}/knn_{k}.xnet"
    g = xn.load(file)
    entries = []

    for res in resolutions:

        #leiden unweighted
        propertyName="Leiden_unweighted_%f"%res
        membership_list=g.vs[propertyName]
        n_clusters=len(set(membership_list))
        results.append({'k': k, 'resolution': res, 'n_clusters': n_clusters})
        #print(f"k={k}, resolution={res}, n_clusters={n_clusters}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Plotting: one subplot per resolution
n_rows = (len(resolutions) + 1) // 2
fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), sharex=False, sharey=False)
axes = axes.flatten()

for i, res in enumerate(resolutions):
    ax = axes[i]
    df_res = results_df[results_df['resolution'] == res]

    # Plot the line
    sns.lineplot(data=df_res, x='k', y='n_clusters', marker='o', ax=ax)

    # Set titles and labels
    ax.set_title(f"Resolution = {res}")
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("# of Clusters")

    # Dynamically set y-axis limits
    min_y = df_res['n_clusters'].min()
    max_y = df_res['n_clusters'].max()
    margin = max(1, int(0.05 * (max_y - min_y)))  # Add a small margin
    ax.set_ylim(min_y - margin, max_y + margin)

# Remove any empty subplots
# for j in range(len(resolutions), len(axes)):
#     fig.delaxes(axes[j])

plt.tight_layout()

plt.savefig("find_best_k.png", dpi=300)
