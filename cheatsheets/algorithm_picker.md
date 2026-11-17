# Which algorithm should I use?

| Situation | Try first | Then |
|---|---|---|
| Tabular, need a baseline fast | Linear/Logistic Regression | Random Forest |
| Tabular, max accuracy | XGBoost / LightGBM | Tuned ensembles |
| Small data, interpretable | Decision Tree, Logistic Regression | Naive Bayes |
| Text classification baseline | TF-IDF + Naive Bayes / Logistic | Fine-tune transformer |
| Non-linear, small/medium data | SVM (RBF) | MLP |
| No labels, know k | K-Means | Gaussian Mixture |
| No labels, unknown k / noise / odd shapes | DBSCAN | Hierarchical |
| Images / audio / text at scale | Deep learning (CNN / Transformer) | Transfer learning |

**Scaling needed?** Yes for KNN, SVM, K-Means, DBSCAN, MLP, regularised linear models. No for tree-based models.
