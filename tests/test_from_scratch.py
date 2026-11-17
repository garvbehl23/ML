"""Every from-scratch model is checked against its scikit-learn counterpart."""
import numpy as np
import pytest
from sklearn import datasets, linear_model, neighbors, naive_bayes, cluster, tree, decomposition
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import from_scratch as fs


def split(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te), y_tr, y_te


def test_linear_regression_matches_sklearn():
    X, y = datasets.make_regression(200, 5, noise=5, random_state=0)
    a, b = fs.LinearRegression().fit(X, y), linear_model.LinearRegression().fit(X, y)
    assert np.allclose(a.coef_, b.coef_) and np.isclose(a.intercept_, b.intercept_)


def test_linear_regression_gd_converges():
    X, y = datasets.make_regression(200, 3, noise=1, random_state=1)
    a = fs.LinearRegression("gd", lr=0.1, n_iter=3000).fit(X, y)
    assert np.allclose(a.coef_, linear_model.LinearRegression().fit(X, y).coef_, atol=1e-2)


def test_logistic_regression_accuracy():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = split(X, y)
    ours = (fs.LogisticRegression(lr=0.1, n_iter=3000).fit(Xtr, ytr).predict(Xte) == yte).mean()
    ref = linear_model.LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte)
    assert abs(ours - ref) < 0.02


def test_knn_matches_sklearn():
    X, y = datasets.load_iris(return_X_y=True)
    Xtr, Xte, ytr, _ = split(X, y)
    assert (fs.KNN(5).fit(Xtr, ytr).predict(Xte) == neighbors.KNeighborsClassifier(5).fit(Xtr, ytr).predict(Xte)).mean() > 0.95


def test_gaussian_nb_matches_sklearn():
    X, y = datasets.load_wine(return_X_y=True)
    Xtr, Xte, ytr, _ = split(X, y)
    assert (fs.GaussianNB().fit(Xtr, ytr).predict(Xte) == naive_bayes.GaussianNB().fit(Xtr, ytr).predict(Xte)).all()


def test_kmeans_recovers_blobs():
    X, y = datasets.make_blobs(300, centers=3, cluster_std=0.6, random_state=0)
    assert adjusted_rand_score(y, fs.KMeans(3).fit(X).labels_) > 0.95


def test_decision_tree_matches_sklearn_accuracy():
    X, y = datasets.load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = split(X, y)
    ours = (fs.DecisionTreeClassifier(max_depth=4).fit(Xtr, ytr).predict(Xte) == yte).mean()
    ref = tree.DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    assert abs(ours - ref) < 0.06


def test_pca_matches_sklearn():
    X, _ = datasets.load_iris(return_X_y=True)
    a, b = fs.PCA(2).fit(X), decomposition.PCA(2).fit(X)
    assert np.allclose(a.explained_variance_ratio_, b.explained_variance_ratio_)
    assert np.allclose(np.abs(a.components_), np.abs(b.components_))
