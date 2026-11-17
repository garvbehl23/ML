"""Cross-validated benchmark of the classic classifiers on three public datasets.

    python benchmarks/run_benchmark.py

Writes benchmarks/results.md and assets/benchmark.png.
"""
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "Breast cancer": datasets.load_breast_cancer(return_X_y=True),
    "Wine": datasets.load_wine(return_X_y=True),
    "Digits": datasets.load_digits(return_X_y=True),
}
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(5),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF)": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(200, random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    "MLP": MLPClassifier((64,), max_iter=800, random_state=0),
}

cv = StratifiedKFold(5, shuffle=True, random_state=0)
acc = {m: {} for m in MODELS}
for dname, (X, y) in DATASETS.items():
    for mname, model in MODELS.items():
        t = time.perf_counter()
        s = cross_val_score(make_pipeline(StandardScaler(), model), X, y, cv=cv)
        acc[mname][dname] = (s.mean(), s.std(), time.perf_counter() - t)

names = list(DATASETS)
lines = ["# Benchmark results", "",
         "5-fold stratified CV accuracy (mean ± std), features standardised. "
         "Regenerate with `python benchmarks/run_benchmark.py`.", "",
         "| Model | " + " | ".join(names) + " | Mean |", "|---|" + "---|" * (len(names) + 1)]
rows = sorted(MODELS, key=lambda m: -np.mean([acc[m][d][0] for d in names]))
for m in rows:
    avg = np.mean([acc[m][d][0] for d in names])
    cells = [f"{acc[m][d][0]:.3f} ± {acc[m][d][1]:.3f}" for d in names]
    lines.append(f"| {m} | " + " | ".join(cells) + f" | **{avg:.3f}** |")
(ROOT / "benchmarks" / "results.md").write_text("\n".join(lines) + "\n")

fig, ax = plt.subplots(figsize=(9, 5))
w = 0.8 / len(names)
for i, d in enumerate(names):
    ax.bar(np.arange(len(rows)) + i * w, [acc[m][d][0] for m in rows], w, label=d)
ax.set_xticks(np.arange(len(rows)) + w)
ax.set_xticklabels(rows, rotation=30, ha="right")
ax.set_ylim(0.7, 1.0)
ax.set_ylabel("CV accuracy")
ax.set_title("Classic classifiers, 5-fold CV")
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "assets" / "benchmark.png", dpi=150)
print("\n".join(lines))
