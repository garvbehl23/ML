import numpy as np


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


class LogisticRegression:
    """Binary logistic regression, L2-regularised, trained with gradient descent."""

    def __init__(self, lr=0.1, n_iter=2000, l2=0.0):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.coef_, self.intercept_ = np.zeros(X.shape[1]), 0.0
        for _ in range(self.n_iter):
            p = sigmoid(X @ self.coef_ + self.intercept_)
            self.coef_ -= self.lr * (X.T @ (p - y) / len(X) + self.l2 * self.coef_)
            self.intercept_ -= self.lr * np.mean(p - y)
        return self

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, float) @ self.coef_ + self.intercept_)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
