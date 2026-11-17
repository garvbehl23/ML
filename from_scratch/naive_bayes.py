import numpy as np


class GaussianNB:
    """Gaussian Naive Bayes, computed in log space for numerical stability."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        self.mu_ = np.array([X[y == c].mean(0) for c in self.classes_])
        eps = self.var_smoothing * X.var(0).max()
        self.var_ = np.array([X[y == c].var(0) for c in self.classes_]) + eps
        return self

    def predict(self, X):
        X = np.asarray(X, float)[:, None, :]
        ll = -0.5 * (np.log(2 * np.pi * self.var_) + (X - self.mu_) ** 2 / self.var_).sum(-1)
        return self.classes_[(ll + np.log(self.prior_)).argmax(1)]
