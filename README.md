<div align="center">

# 🧠 ML — The Ultimate Machine Learning Resource Hub

**Hands-on notebooks · curated learning paths · cheatsheets · the best books, courses & papers**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ready-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

</div>

---

## 📚 Table of Contents
- [What's inside](#-whats-inside)
- [Quick start](#-quick-start)
- [Algorithm notebooks](#-algorithm-notebooks)
- [Learning roadmap](#-learning-roadmap)
- [Curated resources](#-curated-resources)
- [Cheatsheets](#-cheatsheets)
- [Contributing](#-contributing)

## 🔥 What's inside

| Section | Description |
|---|---|
| [`ML libraries.ipynb`](ML%20libraries.ipynb) | NumPy, SciPy, pandas, matplotlib crash course |
| [`Supervised_Learning/`](Supervised_Learning) | 10 notebooks, from linear regression to XGBoost |
| [`Unsupervised Learning/`](Unsupervised%20Learning) | Clustering: K-Means, DBSCAN, Hierarchical |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Zero-to-job-ready ML roadmap |
| [`docs/RESOURCES.md`](docs/RESOURCES.md) | Hand-picked books, courses, papers, tools |
| [`docs/PROJECTS.md`](docs/PROJECTS.md) | Portfolio project ideas by difficulty |
| [`docs/INTERVIEW.md`](docs/INTERVIEW.md) | ML interview question bank |
| [`cheatsheets/`](cheatsheets) | Algorithm picker, metrics, scikit-learn workflow |

## 🚀 Quick start

```bash
git clone https://github.com/garvbehl23/ML.git
cd ML
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## 🧪 Algorithm notebooks

### Supervised learning
| Notebook | Type | Key idea |
|---|---|---|
| [Linear Regression](Supervised_Learning/Linear_Regression.ipynb) | Regression | Fit a line by minimising squared error |
| [Logistic Regression](Supervised_Learning/Logistic_Regression.ipynb) | Classification | Sigmoid over a linear model |
| [KNN](Supervised_Learning/KNN.ipynb) | Both | Predict from the k nearest points |
| [Naive Bayes](Supervised_Learning/Naive_Bayes.ipynb) | Classification | Bayes' rule + independence assumption |
| [SVM](Supervised_Learning/SVM.ipynb) | Both | Maximum-margin hyperplane, kernels |
| [Decision Trees](Supervised_Learning/Decision_Trees.ipynb) | Both | Recursive splits on feature thresholds |
| [Random Forest](Supervised_Learning/Random_Forest.ipynb) | Both | Bagged trees, decorrelated |
| [Gradient Boosting](Supervised_Learning/Gradient_Boosting.ipynb) | Both | Sequentially fit residuals |
| [XGBoost](Supervised_Learning/XGBoost.ipynb) | Both | Regularised, fast gradient boosting |
| [MLP](Supervised_Learning/MLP.ipynb) | Both | Feed-forward neural network |

### Unsupervised learning
| Notebook | Key idea |
|---|---|
| [K-Means](Unsupervised%20Learning/K-means_clustering.ipynb) | Centroid-based partitioning |
| [DBSCAN](Unsupervised%20Learning/DBSCAN.ipynb) | Density-based, finds arbitrary shapes + noise |
| [Hierarchical](Unsupervised%20Learning/Hierarchial_Clustering.ipynb) | Dendrogram-based merging |

## 🗺 Learning roadmap
Math → Python → classical ML → deep learning → specialisation → deployment.
Full week-by-week plan in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## 🌟 Curated resources
Books, courses, papers, datasets and tools: [`docs/RESOURCES.md`](docs/RESOURCES.md).

## 📝 Cheatsheets
- [Which algorithm should I use?](cheatsheets/algorithm_picker.md)
- [Evaluation metrics](cheatsheets/metrics.md)
- [scikit-learn workflow](cheatsheets/sklearn_workflow.md)

## 🤝 Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md). New notebooks, resources and fixes are welcome.

## 📄 License
[MIT](LICENSE)

⭐ If this helps you, star the repo!
