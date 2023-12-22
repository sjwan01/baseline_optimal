import optuna
from optuna import Trial
from typing import List, Type
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from category_encoders import WOEEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def _instantiate_numerical_imputer(trial: Trial) -> SimpleImputer:
    """
    Instantiate and return a numerical imputer based on the value of the
    'numerical_imputer_strategy' parameter suggested by an Optuna trial.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - SimpleImputer: An instance of the SimpleImputer class for numerical features.
    """
    strategy = trial.suggest_categorical(
        # 'most_frequent', 'constant'
        'numerical_imputer_strategy', ['mean', 'median']
    )
    return SimpleImputer(strategy=strategy, fill_value=-1)

def _instantiate_categorical_imputer(trial: Trial) -> SimpleImputer:
    """
    Instantiate and return a categorical imputer based on the value of the
    'categorical_imputer_strategy' parameter suggested by an Optuna trial.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - SimpleImputer: An instance of the SimpleImputer class for categorical features.
    """
    strategy = trial.suggest_categorical(
        'categorical_imputer_strategy', ['most_frequent', 'constant']
    )
    return SimpleImputer(strategy=strategy, fill_value='missing')

def _instantiate_standard_scaler(trial: Trial) -> StandardScaler:
    """
    Instantiate and return a StandardScaler.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - StandardScaler: An instance of the StandardScaler class.
    """
    params = {
        # 'with_mean': trial.suggest_categorical('with_mean', [True, False]),
        # 'with_std': trial.suggest_categorical('with_std', [True, False])
    }
    return StandardScaler(**params)

def _instantiate_min_max_scaler(trial: Trial) -> MinMaxScaler:
    """
    Instantiate and return a MinMaxScaler.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - MinMaxScaler: An instance of the MinMaxScaler class.
    """
    params = {
        # 'clip': trial.suggest_categorical('clip', [True, False])
    }
    return MinMaxScaler(**params)

# def _instantiate_max_abs_scaler(trial):
#     return MaxAbsScaler()

# def _instantiate_robust_scaler(trial):
#     params = {
#         'with_centering': trial.suggest_categorical('with_centering', [True, False]),
#         'with_scaling': trial.suggest_categorical('with_scaling', [True, False]),
#         'unit_variance': trial.suggest_categorical('unit_variance', [True, False])
#     }
#     return RobustScaler(**params)

def _instantiate_scaler(trial: Trial) -> Type[TransformerMixin]:
    """
    Instantiate and return a scaler based on the value of the 'scaler'
    parameter suggested by an Optuna trial.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - Type[TransformerMixin]: An instance of a scaler class.
    """
    scaler = trial.suggest_categorical(
        # 'maxabs', 'robust'
        'scaler', ['standard', 'minmax']
    )
    if scaler == 'standard':
        return _instantiate_standard_scaler(trial)
    elif scaler == 'minmax':
        return _instantiate_min_max_scaler(trial)
    # elif scaler == 'maxabs':
    #     return _instantiate_max_abs_scaler(trial)
    # elif scaler == 'robust':
    #     return _instantiate_max_abs_scaler(trial)

def _instantiate_one_hot_encoder(trial: Trial) -> OneHotEncoder:
    """
    Instantiate and return a OneHotEncoder.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - OneHotEncoder: An instance of the OneHotEncoder class.
    """
    return OneHotEncoder(drop='first')

def _instantiate_ordinal_encoder(trial: Trial) -> OrdinalEncoder:
    """
    Instantiate and return an OrdinalEncoder.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - OrdinalEncoder: An instance of the OrdinalEncoder class.
    """
    return OrdinalEncoder()

# def _instantiate_target_encoder(trial):
#     return TargetEncoder()

# def _instantiate_woe_encoder(trial):
#     return WOEEncoder()

def _instantiate_encoder(trial: Trial) -> Type[TransformerMixin]:
    """
    Instantiate and return an encoder based on the value of the 'encoder'
    parameter suggested by an Optuna trial.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - Type[TransformerMixin]: An instance of an encoder class.
    """
    encoder = trial.suggest_categorical(
        # 'target', 'woe'
        'encoder', ['onehot', 'ordinal']
    )
    if encoder == 'onehot':
        return _instantiate_one_hot_encoder(trial)
    elif encoder == 'ordinal':
        return _instantiate_ordinal_encoder(trial)
    # elif encoder == 'target':
    #     return _instantiate_target_encoder(trial)
    # elif encoder == 'woe':
    #     return _instantiate_woe_encoder(trial)

def _instantiate_numerical_pipeline(trial: Trial) -> Pipeline:
    """
    Instantiate and return a pipeline for numerical features.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - Pipeline: An instance of the scikit-learn Pipeline class.
    """
    pipeline = Pipeline([
        ('numerical_imputer', _instantiate_numerical_imputer(trial)),
        ('scaler', _instantiate_scaler(trial))
    ])
    return pipeline

def _instantiate_categorical_pipeline(trial: Trial) -> Pipeline:
    """
    Instantiate and return a pipeline for categorical features.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
    - Pipeline: An instance of the scikit-learn Pipeline class.
    """
    pipeline = Pipeline([
        ('categorical_imputer', _instantiate_categorical_imputer(trial)),
        ('encoder', _instantiate_encoder(trial))
    ])
    return pipeline

def _feature_selector(trial: Trial, features: List[str]) -> List[str]:
    """
    Perform feature selection based on Optuna trial suggestions.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.
    - features (List[str]): List of features.

    Returns:
    - List[str]: List of selected features.
    """
    select = lambda feature: trial.suggest_categorical(feature, [True, False])
    selected = [*filter(select, features)]
    return selected

def _instantiate_processor(trial: Trial, numerical_features: List[str],
 categorical_features: List[str], select: bool=False) -> ColumnTransformer:
    """
    Instantiate and return a ColumnTransformer for preprocessing.

    Args:
    - trial (optuna.trial.Trial): The Optuna trial object.
    - numerical_features (List[str]): List of numerical features.
    - categorical_features (List[str]): List of categorical features.
    - select (bool): Whether to perform feature selection.

    Returns:
    - ColumnTransformer: An instance of the scikit-learn ColumnTransformer class.
    """
    numerical_pipeline = _instantiate_numerical_pipeline(trial)
    categorical_pipeline = _instantiate_categorical_pipeline(trial)
    if select:
        numerical_features = _feature_selector(trial, numerical_features)
        numerical_features = _feature_selector(trial, categorical_features)
    processor = ColumnTransformer([
        ('numerical_pipeline', numerical_pipeline, numerical_features),
        ('categorical_pipeline', categorical_pipeline, categorical_features)
    ])
    return processor
