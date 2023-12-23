# Introduction to Baseline Optimal

In machine learning projects, achieving optimal evaluation metric scores involves navigating through multiple steps, including data cleaning, processing, model selection, and hyperparameter tuning. This iterative process often produces repetitive and error-prone code, resulting in suboptimal outcomes.

The `baseline_optimal` package automates the process of pipeline tuning by employing [Optuna](https://optuna.readthedocs.io/en/stable/index.html)'s Bayesian optimization, significantly reducing the need for manual experimentation. With the help of Optuna, the package selects and optimizes pipeline components through iterative trials, leading to the discovery of the most effective combination of data preprocessing, model selection, and hyperparameter configurations.

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

---

## Documentation

Access the the entire documentation through [GitHub Pages](https://sjwan01.github.io/baseline_optimal/).

Check out `baseline_optimal` modules available and their respective documentation as well as example.

<div align="center">

| Modules | Task | Documentation | Example |
| - | - |-- | - |
| `baseline_optimal.class_task` | classification | [**Link**](https://sjwan01.github.io/baseline_optimal/class_task.html) | [**Link**]() |

</div>

Check out machine learning algorithms supported and hyperparameters considered.

<div align="center">

| Algorithm | Source | Hyperparameters |
| - | - | - |
| `DecisionTreeClassifier` | `sklearn.tree` | `max_features`<br>`max_depth`<br>`min_samples_split` |
| `RandomForestClassifier` | `sklearn.ensemble` | `n_estimators`<br>`max_features`<br>`max_depth`<br>`min_samples_split` |
| `AdaBoostClassifier` | `sklearn.ensemble` | `n_estimators`<br>`learning_rate` |
| `XGBClassifier` | `xgboost` | `n_estimators`<br>`learning_rate`<br>`max_depth` |

</div>

<!-- ## TODO

- random state config
- classification tasks with imbalanced data
- regression tasks
- trial pruning
- more estimators
- larger hyperparameter space
- more visualizations