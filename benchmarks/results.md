# Benchmark results

5-fold stratified CV accuracy (mean ± std), features standardised. Regenerate with `python benchmarks/run_benchmark.py`.

| Model | Breast cancer | Wine | Digits | Mean |
|---|---|---|---|---|
| SVM (RBF) | 0.977 ± 0.007 | 0.983 ± 0.014 | 0.981 ± 0.007 | **0.980** |
| Logistic Regression | 0.979 ± 0.014 | 0.983 ± 0.014 | 0.969 ± 0.010 | **0.977** |
| Random Forest | 0.967 ± 0.019 | 0.978 ± 0.021 | 0.976 ± 0.005 | **0.973** |
| MLP | 0.975 ± 0.015 | 0.972 ± 0.018 | 0.972 ± 0.011 | **0.973** |
| KNN | 0.965 ± 0.021 | 0.961 ± 0.013 | 0.977 ± 0.006 | **0.967** |
| Gradient Boosting | 0.965 ± 0.018 | 0.949 ± 0.021 | 0.967 ± 0.004 | **0.960** |
| Decision Tree | 0.926 ± 0.022 | 0.927 ± 0.051 | 0.860 ± 0.021 | **0.905** |
| Naive Bayes | 0.930 ± 0.012 | 0.972 ± 0.000 | 0.786 ± 0.032 | **0.896** |
