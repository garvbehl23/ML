# 🗺 ML Roadmap (≈ 6 months, 8–10 h/week)

## Phase 1 — Foundations (weeks 1–4)
- **Python**: functions, classes, comprehensions → [`ML libraries.ipynb`](../ML%20libraries.ipynb)
- **Math**: linear algebra (vectors, matrices, eigenvalues), calculus (gradients, chain rule), probability & statistics (distributions, Bayes, expectation, variance)
- **Goal**: implement gradient descent from scratch in NumPy.

## Phase 2 — Classical ML (weeks 5–12)
Work through the notebooks in this order:
1. Linear Regression → Logistic Regression
2. KNN → Naive Bayes → SVM
3. Decision Trees → Random Forest → Gradient Boosting → XGBoost
4. K-Means → DBSCAN → Hierarchical
- Learn: train/val/test split, cross-validation, bias–variance, regularisation (L1/L2), feature engineering, pipelines, metrics ([cheatsheet](../cheatsheets/metrics.md)).
- **Goal**: finish 2 Kaggle tabular competitions/practice sets.

## Phase 3 — Deep learning (weeks 13–20)
- MLP (see notebook) → backpropagation by hand → PyTorch basics
- CNNs (vision), RNN/LSTM → Transformers (NLP)
- Training tricks: batch norm, dropout, learning-rate schedules, mixed precision
- **Goal**: train an image classifier and fine-tune a pretrained transformer.

## Phase 4 — Specialise (weeks 21–24)
Pick one: NLP/LLMs · Computer Vision · Recommender systems · Time series · Reinforcement learning · MLOps.

## Phase 5 — Ship it
Experiment tracking (MLflow / Weights & Biases), model serving (FastAPI), Docker, monitoring for drift, data versioning (DVC). Build 2–3 portfolio projects: [PROJECTS.md](PROJECTS.md).
