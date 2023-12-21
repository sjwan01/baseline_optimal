import optuna as op
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

def _instantiate_estimator(trial):
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

def _instantiate_decision_tree(trial):
    params = {
        'max_features': trial.suggest_categorical('dt_max_features', ['log2', 'sqrt', None]),
        'max_depth': trial.suggest_int('dt_max_depth', 5, 10),
        'min_samples_split': trial.suggest_int('dt_min_sample_split', 2, 10),
        'random_state': 123
    }
    return DecisionTreeClassifier(**params)

def _instantiate_random_forest(trial):
    params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 100, 300),
        'max_features': trial.suggest_categorical('rf_max_features', ['log2', 'sqrt', None]),
        'max_depth': trial.suggest_int('rf_max_depth', 5, 10),
        'min_samples_split': trial.suggest_int('rf_min_sample_split', 2, 10),
        'random_state': 123
    }
    return RandomForestClassifier(**params)

def _instantiate_adaboost(trial):
    params = {
        'n_estimators': trial.suggest_int('ada_n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('ada_learning_rate', 0.001, 0.1),
        'random_state': 123
    }
    return AdaBoostClassifier(**params)

def _instantiate_xgboost(trial):
    params = {
        'eta': trial.suggest_float('xgb_eta', 0, 0.1),
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 300),
        'max_features': trial.suggest_categorical('xgb_max_features', ['log2', 'sqrt', None]),
        'max_depth': trial.suggest_int('xgb_max_depth', 5, 10),
        'min_samples_split': trial.suggest_int('xgb_min_sample_split', 2, 10),
        'random_state': 123
    }
    return XGBClassifier(**params)