import numpy as np


class LinearRegression:
    """Ordinary least squares via the normal equation (pseudo-inverse, so it
    also works when X^T X is singular), or batch gradient descent."""

    def __init__(self, method="normal", lr=0.01, n_iter=5000):
        self.method, self.lr, self.n_iter = method, lr, n_iter

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        Xb = np.c_[np.ones(len(X)), X]
        if self.method == "normal":
            w = np.linalg.pinv(Xb) @ y
        else:
            w = np.zeros(Xb.shape[1])
            for _ in range(self.n_iter):
                w -= self.lr * (2 / len(Xb)) * Xb.T @ (Xb @ w - y)
        self.intercept_, self.coef_ = w[0], w[1:]
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_
