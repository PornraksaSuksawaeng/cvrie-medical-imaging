"""Plots for the clustering results."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

GREY = "#d9d9d9"


def project_2d(features):
    """Reduce the LSA vectors to 2 dimensions so we can draw them.

    Only used for the picture. We never cluster on these 2D points.
    """
    tsne = TSNE(n_components=2, metric="cosine", init="pca", random_state=42)
    return tsne.fit_transform(features)


def to_plot_color(color):
    """Turn '0x000000' into '#000000'. Missing colours become grey."""
    if pd.isna(color):
        return GREY
    return "#" + color[2:]


def plot_dataset_colors(ax, coords, colors):
    """Each testimony drawn with the colour from the dataset."""
    plot_colors = []
    for color in colors:
        plot_colors.append(to_plot_color(color))

    ax.scatter(coords[:, 0], coords[:, 1], c=plot_colors, s=15)
    ax.set_title("Coloured by the dataset colour column")


def plot_clusters(ax, coords, labels):
    """Each testimony drawn with the colour of its cluster."""
    palette = plt.get_cmap("tab20c")

    for cluster in sorted(set(labels)):
        in_cluster = labels == cluster
        size = in_cluster.sum()
        ax.scatter(
            coords[in_cluster, 0],
            coords[in_cluster, 1],
            color=palette(cluster),
            s=15,
            label=f"cluster {cluster} ({size})",
        )

    ax.set_title("Clusters found by KMeans")
    ax.legend(fontsize=7, ncol=2)
