import optuna as op
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from category_encoders import WOEEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def _instantiate_numerical_imputer(trial):
    strategy = trial.suggest_categorical(
        # 'most_frequent', 'constant'
        'numerical_imputer_strategy', ['mean', 'median']
    )
    return SimpleImputer(strategy=strategy, fill_value=-1)

def _instantiate_categorical_imputer(trial):
    strategy = trial.suggest_categorical(
        'categorical_imputer_strategy', ['most_frequent', 'constant']
    )
    return SimpleImputer(strategy=strategy, fill_value='missing')

def _instantiate_standard_scaler(trial):
    params = {
        # 'with_mean': trial.suggest_categorical('with_mean', [True, False]),
        # 'with_std': trial.suggest_categorical('with_std', [True, False])
    }
    return StandardScaler(**params)

def _instantiate_min_max_scaler(trial):
    params = {
        # 'clip': trial.suggest_categorical('clip', [True, False])
    }
    return MinMaxScaler(**params)

def _instantiate_max_abs_scaler(trial):
    return MaxAbsScaler()

def _instantiate_robust_scaler(trial):
    params = {
        # 'with_centering': trial.suggest_categorical('with_centering', [True, False]),
        # 'with_scaling': trial.suggest_categorical('with_scaling', [True, False]),
        # 'unit_variance': trial.suggest_categorical('unit_variance', [True, False])
    }
    return RobustScaler(**params)

def _instantiate_scaler(trial):
    scaler = trial.suggest_categorical(
        # 'maxabs', 'robust'
        'scaler', ['standard', 'minmax']
    )
    if scaler == 'standard':
        return _instantiate_standard_scaler(trial)
    elif scaler == 'minmax':
        return _instantiate_min_max_scaler(trial)
    elif scaler == 'maxabs':
        return _instantiate_max_abs_scaler(trial)
    elif scaler == 'robust':
        return _instantiate_max_abs_scaler(trial)

def _instantiate_one_hot_encoder(trial):
    return OneHotEncoder(drop='first')

def _instantiate_ordinal_encoder(trial):
    return OrdinalEncoder()

def _instantiate_target_encoder(trial):
    return TargetEncoder()

def _instantiate_woe_encoder(trial):
    return WOEEncoder()

def _instantiate_encoder(trial):
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

def _instantiate_numerical_pipeline(trial):
    pipeline = Pipeline([
        ('numerical_imputer', _instantiate_numerical_imputer(trial)),
        ('scaler', _instantiate_scaler(trial))
    ])
    return pipeline

def _instantiate_categorical_pipeline(trial):
    pipeline = Pipeline([
        ('categorical_imputer', _instantiate_categorical_imputer(trial)),
        ('encoder', _instantiate_encoder(trial))
    ])
    return pipeline

def _feature_selector(trial, features):
    select = lambda feature: trial.suggest_categorical(feature, [True, False])
    selected = [*filter(select, features)]
    return selected

def _instantiate_processor(trial, numerical_features, categorical_features, select=False):
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
