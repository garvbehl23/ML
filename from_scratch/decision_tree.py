import numpy as np


def gini(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return 1 - (p ** 2).sum()


class DecisionTreeClassifier:
    """CART classifier using Gini impurity and axis-aligned threshold splits."""

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth, self.min_samples_split = max_depth, min_samples_split

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.tree_ = self._grow(X, np.searchsorted(self.classes_, y), 0)
        return self

    def _best_split(self, X, y):
        best = (None, None, gini(y))
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j])
            xs, ys = X[order, j], y[order]
            for i in range(1, len(xs)):
                if xs[i] == xs[i - 1]:
                    continue
                g = (i * gini(ys[:i]) + (len(ys) - i) * gini(ys[i:])) / len(ys)
                if g < best[2] - 1e-12:
                    best = (j, (xs[i] + xs[i - 1]) / 2, g)
        return best

    def _grow(self, X, y, depth):
        leaf = {"value": np.bincount(y, minlength=len(self.classes_)).argmax()}
        if (len(np.unique(y)) == 1 or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return leaf
        j, t, _ = self._best_split(X, y)
        if j is None:
            return leaf
        m = X[:, j] <= t
        return {"feature": j, "threshold": t,
                "left": self._grow(X[m], y[m], depth + 1),
                "right": self._grow(X[~m], y[~m], depth + 1)}

    def _one(self, x, node):
        while "value" not in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]

    def predict(self, X):
        return self.classes_[[self._one(x, self.tree_) for x in np.asarray(X, float)]]
