import numpy as np


class KNN:
    """k-nearest-neighbours classifier (Euclidean, majority vote)."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        d = ((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(d, axis=1, kind="stable")[:, : self.k]
        votes = np.stack([(self.y_[idx] == c).sum(1) for c in self.classes_], 1)
        return self.classes_[votes.argmax(1)]
