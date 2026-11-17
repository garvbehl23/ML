# Evaluation metrics

## Classification
- **Accuracy** = (TP+TN)/all — misleading on imbalanced data
- **Precision** = TP/(TP+FP) — "of predicted positives, how many are right"
- **Recall** = TP/(TP+FN) — "of actual positives, how many found"
- **F1** = 2PR/(P+R)
- **ROC-AUC** — ranking quality across thresholds; **PR-AUC** is better when positives are rare
- **Log-loss** — penalises confident wrong probabilities

## Regression
- **MAE** — average absolute error, robust to outliers
- **MSE / RMSE** — penalises large errors
- **R²** — fraction of variance explained

## Clustering
- **Silhouette score** (-1 to 1, higher is better)
- **Davies–Bouldin** (lower is better)
- **Adjusted Rand Index** (when ground truth exists)
