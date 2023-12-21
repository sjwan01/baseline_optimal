import numpy as np
import pandas as pd
import optuna as op
import shap as sp
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from ._processor import _instantiate_processor
from ._classifier import _instantiate_estimator


class ClassificationStudy:

    def __init__(self):
        self.study = None
        self.best_pipeline = None
        self.best_score = None

    def _instantiate_pipeline(trial, numerical_features, categorical_features, select=False):
        processor = _instantiate_processor(
            trial, numerical_features, categorical_features, select
        )
        estimator = _instantiate_estimator(trial)
        pipeline = Pipeline([
            ('processor', processor),
            ('estimator', estimator)
        ])
        return pipeline

    def optimize(self, trial, X, y, metric='accuracy', select=False, study_name='optimization', cv=5, n_trials=100):

        numerical_features = [
            *X.select_dtypes(exclude=['object', 'category']).columns
        ]
        categorical_features = [
            *X.select_dtypes(include=['object', 'object']).columns
        ]

        def objective(self):
            pipeline = self._instantiate_pipeline(
                trial, numerical_features, categorical_features, select
            )
            scores = cross_val_score(pipeline, X, y, scoring=metric, cv=cv)
            return np.min([np.mean(scores), np.median([scores])])

        op.logging.set_verbosity(op.logging.WARNING)
        self.study = op.create_study(study_name=study_name, direction='maximize')
        self.study.optimize(lambda trial: objective(self, trial, X, y), n_trials=n_trials)
        self.best_pipeline = self._instantiate_pipeline(self.study.best_trial, numerical_features, categorical_features, select)

    def fit(self, X, y):
        self.best_pipeline.fit(X, y)

    def evaluate(self, X, y, threshold=0.5):
        pred_y_proba = self.best_pipeline.predict_proba(X)[:, 1]
        scores = {
            'AUC-ROC': roc_auc_score(y, pred_y_proba),
            'Accuracy': accuracy_score(y, pred_y_proba >= threshold),
            'Average Precision': average_precision_score(y, pred_y_proba),
            'Recall': recall_score(y, pred_y_proba >= threshold),
            'Precision': precision_score(y, pred_y_proba >= threshold),
            'F1': f1_score(y, pred_y_proba >= threshold)
        }
        return pred_y_proba, pd.DataFrame(scores).iloc[0,:].to_frame().T.round(4)

    def plot_optimization_history(self):
        return op.visualization.plot_optimization_history(self.study)

    def plot_param_importances(self):
        return op.visualization.plot_param_importances(self.study)

    def plot_feature_importances(self, X):
        explainer = sp.Explainer(self.best_pipeline[-1])
        X_transformed = self.best_pipeline[0].transform(X)
        feature_names = [feature.split('__')[1] for feature in self.best_pipeline[0].get_feature_names_out()]
        shap_values = explainer.shap_values(X_transformed)
        return sp.summary_plot(shap_values, X_transformed, feature_names)
    
    @property
    def best_pipeline(self):
        return self.best_pipeline

    @property
    def best_score(self):
        return self.best_score