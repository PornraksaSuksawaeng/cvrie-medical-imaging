"""Clustering models and the label-free metrics used to compare them."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

RANDOM_STATE = 42


def fit_kmeans(features, n_clusters: int, random_state: int = RANDOM_STATE):
    """Minimise inertia: the sum of squared distances to each cluster centroid."""
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    return model.fit_predict(features)


def fit_agglomerative(features, n_clusters: int, linkage: str = "ward"):
    """Bottom-up merging of the nearest pairs until n_clusters remain."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(features)


def score_clustering(features, labels) -> dict:
    """Internal validation indices. No ground truth involved."""
    real_clusters = set(labels) - {-1}
    if len(real_clusters) < 2:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }
    return {
        "silhouette": silhouette_score(features, labels, metric="cosine"),
        "davies_bouldin": davies_bouldin_score(features, labels),
        "calinski_harabasz": calinski_harabasz_score(features, labels),
    }


def compare_partitions(labels_a, labels_b) -> float:
    """Adjusted Rand index between two clusterings."""
    return adjusted_rand_score(labels_a, labels_b)


def top_terms_per_cluster(matrix, labels, feature_names, n_terms: int = 8) -> dict:
    """Average TF-IDF weight per term inside each cluster"""
    summary = {}
    for label in sorted(set(labels)):
        mask = labels == label
        centroid = np.asarray(matrix[mask].mean(axis=0)).ravel()
        top = centroid.argsort()[::-1][:n_terms]
        summary[int(label)] = [feature_names[i] for i in top]
    return summary


def kmeans_inertia(
    features, n_clusters: int, random_state: int = RANDOM_STATE
) -> float:
    """The value KMeans minimises: sum of squared distances to the nearest centre."""
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    model.fit(features)
    return model.inertia_
