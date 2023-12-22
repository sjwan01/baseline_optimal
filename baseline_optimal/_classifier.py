import optuna
from optuna import Trial
from typing import Type
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

def _instantiate_estimator(trial: Trial) -> Type[ClassifierMixin]:
    """
    Instantiate and return a classifier based on the value of the 'estimator'
    parameter suggested by an Optuna trial.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - Type[ClassifierMixin]: An instance of a classifier class.
    """
    estimator = trial.suggest_categorical(
        'estimator', ['decision_tree', 'random_forest', 'adaboost', 'xgboost']
    )
    if estimator == 'decision_tree':
        return _instantiate_decision_tree(trial)
    elif estimator == 'random_forest':
        return _instantiate_random_forest(trial)
    elif estimator == 'adaboost':
        return _instantiate_adaboost(trial)
    elif estimator == 'xgboost':
        return _instantiate_xgboost(trial)

def _instantiate_decision_tree(trial: Trial) -> DecisionTreeClassifier:
    """
    Instantiate and return a Decision Tree classifier based on hyperparameters
    suggested by an Optuna trial.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - DecisionTreeClassifier: An instance of the DecisionTreeClassifier class.
    """
    params = {
        'max_features': trial.suggest_categorical('dt_max_features', ['log2', 'sqrt', None]),
        'max_depth': trial.suggest_int('dt_max_depth', 5, 10),
        'min_samples_split': trial.suggest_int('dt_min_sample_split', 2, 10),
        'random_state': 123
    }
    return DecisionTreeClassifier(**params)

def _instantiate_random_forest(trial: Trial) -> RandomForestClassifier:
    """
    Instantiate and return a Random Forest classifier based on hyperparameters
    suggested by an Optuna trial.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - RandomForestClassifier: An instance of the RandomForestClassifier class.
    """
    params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 100, 300),
        'max_features': trial.suggest_categorical('rf_max_features', ['log2', 'sqrt', None]),
        'max_depth': trial.suggest_int('rf_max_depth', 5, 10),
        'min_samples_split': trial.suggest_int('rf_min_sample_split', 2, 10),
        'random_state': 123
    }
    return RandomForestClassifier(**params)

def _instantiate_adaboost(trial: Trial) -> AdaBoostClassifier:
    """
    Instantiate and return an AdaBoost classifier based on hyperparameters
    suggested by an Optuna trial.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - AdaBoostClassifier: An instance of the AdaBoostClassifier class.
    """
    params = {
        'n_estimators': trial.suggest_int('ada_n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('ada_learning_rate', 0.001, 0.1),
        'random_state': 123
    }
    return AdaBoostClassifier(**params)

def _instantiate_xgboost(trial: Trial) -> XGBClassifier:
    """
    Instantiate and return an XGBoost classifier based on hyperparameters
    suggested by an Optuna trial.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - XGBClassifier: An instance of the XGBClassifier class.
    """
    params = {
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0, 0.1),
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 300),
        'max_depth': trial.suggest_int('xgb_max_depth', 5, 10),
        'random_state': 123
    }
    return XGBClassifier(**params)