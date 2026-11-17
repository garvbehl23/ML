import numpy as np


class PCA:
    """PCA via SVD of the centred data matrix."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(0)
        _, s, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        var = s ** 2 / (len(X) - 1)
        self.explained_variance_ = var[: self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / var.sum()
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
