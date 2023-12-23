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

Check out `baseline_optimal` modules available and their respective documentation as well as example.

| Modules | Task | Documentation | Example |
| - | - |-- | - |
| `baseline_optimal.class_task` | classification | [**Link**]() | [**Link**]() |


<!-- ## TODO

- classification tasks with imbalanced data
- regression tasks
- trial pruning
- more estimators
- larger hyperparameter space
- more visualizations