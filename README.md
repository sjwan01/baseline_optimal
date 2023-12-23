# Introduction to Baseline Optimal

In machine learning projects, achieving optimal evaluation metric scores involves navigating through multiple steps, including data cleaning, processing, model selection, and hyperparameter tuning. This iterative process often produces repetitive and error-prone code, resulting in suboptimal outcomes.

The `baseline_optimal` package automates the process of pipeline tuning by employing [Optuna](https://optuna.readthedocs.io/en/stable/index.html)'s Bayesian optimization, significantly reducing the need for manual experimentation. With the help of Optuna, the package intelligently selects and optimizes pipeline components through iterative trials, leading to the discovery of the most effective combination of data preprocessing, model selection, and hyperparameter configurations.

___

## Installation

You can install the `baseline_optimal` package and its dependencies using `pip`:

```bash
pip install baseline_optimal
```

After installation, you can import the package in Python:

```python
import baseline_optimal
```

## Usage

The current version supports binary classification tasks with balanced data.

To prepare your data, make sure to:
1. Remove columns that make no sense to machine learning models (e.g. name, email, text).
2. Split the training and the test set, and encode the target variable if necessary.

To start with, declare a `ClassTask` object:

```python
from baseline_optimal import ClassTask

task = ClassTask()
```

Search through the parameter space and optimize for the accuracy score by default. Other supported metrics for classification tasks can be found through [`sklearn`](https://scikit-learn.org/stable/modules/model_evaluation.html). Set `select=True` when the dataset is large to enable feature selection prior to fitting models. `cv` defines the number of folds within each iteration, and it can also take a [CV splitter](https://scikit-learn.org/stable/modules/cross_validation.html). The `optimize` function aims to discover the optimal machine learning pipeline, incorporating the best data processor and estimator with optimized hyperparameters. Subsequently, it fits this optimal pipeline to the training data.

```python
task.optimize(X = X_train, y = y_train, metric = 'accuracy', select = False,
              study_name = 'optimization', cv = 5, n_trials = 100)
```

Access the best CV score.

```python
print(task.best_score)
```

After fitting the data, one option is to use the `ClassTask` object for probability prediction and evaluation on the test set. You may customize the classification threshold.

```python
pred_prob = study.predict(X = X_test)

scores = study.evaluate(X = X_test, y_true = y_test, threshold = 0.5)
```

Or, you may obtain the optimal [`imblearn.pipeline.Pipeline`](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html) object.

```python
best_pipeline = study.best_pipeline

pred_prob = best_pipeline.predict_proba(X_test)

best_processor = best_pipeline.named_steps['processor']
best_estimator = best_pipeline.named_steps['estimator']
```

You may access the sketchy documentation [here](https://sjwan01.github.io/baseline_optimal/) to see what can be visualized.

---

## TODO

The upcoming version of the `baseline_optimal` package will support:

- classification tasks with imbalanced data
- regression tasks
- trial pruning
- more estimators
- larger hyperparameter space
- more visualizations
- more feature transformation
- dimensionality reduction