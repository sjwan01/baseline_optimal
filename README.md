# Introduction to Baseline Optimal

Between the raw data and the optimal results in machine learning projects, there is an exhausting, iterative process. We go back and forth to experiment with various combinations of feature engineering, processing methods, and models along with their hyperparameters. At the end of the day, we hope our efforts pay off.

🤞 God bless data scientists. 🤞

Manual experimentation is a good practice. However, you may have found that we often produce messy, repetitive code throughout the process, and it takes a long while for us to figure out that an attempt doesn't work out. Sometimes we may overcomplicate data transformation and processing to get promising but unncessary metric scores.

Given these problems, the `baseline_optimal` package automates the workflow by employing [Optuna](https://optuna.readthedocs.io/en/stable/index.html)'s Bayesian optimization, significantly reducing the need for manual experimentation. You provide the raw data, and the modules do the heavy lifting. 

---

## Installation

You can install the `baseline_optimal` package and its dependencies using `pip`:

```bash
pip install baseline_optimal
```

After installation, you can import the package in Python:

```python
import baseline_optimal
```
___

## Documentation

Access the the entire documentation through [GitHub Pages](https://sjwan01.github.io/baseline_optimal/).

Check out `baseline_optimal` modules available and their respective documentation as well as example.

<div align="center">

| Modules | Task | Documentation | Example |
| - | - |-- | - |
| `baseline_optimal.class_task` | classification | [**Link**](https://sjwan01.github.io/baseline_optimal/class_task.html) | [**Link**](https://sjwan01.github.io/baseline_optimal/class_task_example.html) |

</div>

Check out machine learning algorithms supported and hyperparameters considered.

<div align="center">

| Algorithm | Source | Hyperparameters |
| - | - | - |
| [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) | `sklearn.tree` | `max_features`<br>`max_depth`<br>`min_samples_split` |
| [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | `sklearn.ensemble` | `n_estimators`<br>`max_features`<br>`max_depth`<br>`min_samples_split` |
| [`AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) | `sklearn.ensemble` | `n_estimators`<br>`learning_rate` |
| [`XGBClassifier`](https://xgboost.readthedocs.io/en/stable/python/python_api.html) | `xgboost` | `n_estimators`<br>`learning_rate`<br>`max_depth` |

</div>

---

## Why "Baseline" Optimal

The current version supports feature selection, missing value imputation, scaling and encoding as data transformation and processing steps. The pipeline performance is evaluated based on choices of these components along with multiple machine learning algorithms. With help of Optuna, the package gives you the optimal workflow provided the raw data.

The results are "baseline" optimal because the workflow attempts only the most basic methods. No feature engineering or dimensionality reduction, so on and so forth. It aims to answer the lazy question that, "If I do nothing, how far can I get?" By using this package, if you get satisfting results then congradulations! If not, then you know where the baseline is and you might want to do better than that based on your domain knowledge.

🤞 Good luck. 🤞

<!-- ## TODO

- random state config
- classification tasks with imbalanced data
- regression tasks
- trial pruning
- more estimators
- larger hyperparameter space
- more visualizations