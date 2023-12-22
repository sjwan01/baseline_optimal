# Introduction to Baseline Optimal

In machine learning projects, achieving optimal evaluation metric scores involves navigating through multiple steps, including data cleaning, processing, model selection, and hyperparameter tuning. This iterative process often produces repetitive and error-prone code, resulting in suboptimal outcomes.

The `baseline_optimal` package automates the process of hyperparameter tuning by employing [Optuna](https://optuna.readthedocs.io/en/stable/index.html)'s Bayesian optimization, significantly reducing the need for manual experimentation. With the help of Optuna, the package intelligently selects and optimizes pipeline components through iterative trials, leading to the discovery of the most effective combination of data preprocessing, model selection, and hyperparameter configurations.

___

## Installation

To install the `baseline_optimal` package and its dependencies, you can use the following commands:

```bash
pip install baseline_optimal
```

After installation, you can import the package in Python with the following code:

```python
import baseline_optimal
```

## Usage