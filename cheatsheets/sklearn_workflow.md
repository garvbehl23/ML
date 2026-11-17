# scikit-learn workflow template

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42)),
])

print(cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro").mean())
pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))
```

Tuning: `GridSearchCV` / `RandomizedSearchCV` on the pipeline (`model__max_depth`, ...).
Rule: fit preprocessing **only on training data** (the pipeline does this for you) to avoid leakage.
