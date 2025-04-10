import numpy as np
from sklearn.metrics import silhouette_score

def sil_computation(data, labels):
    """
    Compute the Silhouette Score excluding noise points (-1 labels).
    """
    mask = labels != -1  # -1 are noise points
    data_filtered = data[mask]
    labels_filtered = labels[mask]

    if len(np.unique(labels_filtered)) > 1:  # we need at least two clusters
        score = silhouette_score(data_filtered, labels_filtered)
        print(f'Silhouette Score: {score:.2f}')
        return score
    else:
        print('Silhouette score cannot be computed for a single cluster or only noise points.')
        return None