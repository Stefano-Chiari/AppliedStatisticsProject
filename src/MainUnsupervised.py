import pandas as pd
import numpy as np

from Unsupervised.KMeans import KMeans

class Unsupervised:
    """
    Handles Dimensionality Reduction and Clustering.
    Focuses on structure discovery without a target variable.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.pca_results = None
        self.cluster_labels = None

    def run_pca(self, n_components=2):
        """Linear dimensionality reduction."""
        pass

    def run_nonlinear_reduction(self, method='tsne'):
        """t-SNE or UMAP implementation."""
        pass

    def run_clustering(self, algorithm='kmeans', n_clusters=3):
        """K-Means or Hierarchical clustering."""
        # Example implementation, the class is still not working
        if algorithm == 'kmeans':
            model = KMeans()

    def visualize_latent_space(self):
        """Plotting components colored by clusters."""
        pass