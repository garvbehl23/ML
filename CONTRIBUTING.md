# Contributing

1. Fork the repo and create a branch (`git checkout -b add-lightgbm-notebook`).
2. Notebooks: clear outputs of heavy cells, start with a markdown intro (theory, when to use, pitfalls), use a public dataset or `sklearn.datasets`, and end with an evaluation section.
3. Resources: only add things you have actually used. Include a one-line reason it's good and whether it's free.
4. Keep file names `Title_Case_With_Underscores.ipynb`.
5. Open a pull request describing what you added.
7. For anything under `from_scratch/`, add a test in `tests/` comparing it with the scikit-learn equivalent, and run `python -m pytest -q tests` before opening a PR.
