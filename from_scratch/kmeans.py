import numpy as np


class KMeans:
    """Lloyd's algorithm with k-means++ seeding."""

    def __init__(self, k=3, n_iter=300, tol=1e-6, seed=0):
        self.k, self.n_iter, self.tol, self.seed = k, n_iter, tol, seed

    def _init(self, X, rng):
        c = [X[rng.integers(len(X))]]
        for _ in range(self.k - 1):
            d = np.min([((X - ci) ** 2).sum(1) for ci in c], axis=0)
            c.append(X[rng.choice(len(X), p=d / d.sum())])
        return np.array(c)

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        C = self._init(X, rng)
        for _ in range(self.n_iter):
            labels = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            new = np.array([X[labels == j].mean(0) if (labels == j).any() else C[j]
                            for j in range(self.k)])
            shift = np.abs(new - C).max()
            C = new
            if shift < self.tol:
                break
        self.centers_, self.labels_ = C, labels
        self.inertia_ = ((X - C[labels]) ** 2).sum()
        return self

    def predict(self, X):
        return ((np.asarray(X, float)[:, None] - self.centers_[None]) ** 2).sum(-1).argmin(1)
