import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna as op
import shap as sp
from optuna import Trial
from typing import List, Tuple, Union, Iterable
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from ._processor import _instantiate_processor
from ._classifier import _instantiate_estimator


class ClassTask:
    """
        Class for optimizing machine learning pipelines using Bayesian optimization.

        Attributes:
            _study (optuna.study.Study): Optuna study object containing the optimization results.
            _best_pipeline (sklearn.pipeline.Pipeline): Best machine learning pipeline obtained from optimization.
            _best_score (float): Best score achieved during optimization.

        Methods:
            optimize(X, y, metric='accuracy', select=False, study_name='optimization', cv=5, n_trials=100):
                Optimize the machine learning pipeline using Bayesian optimization.

            predict(X):
                Predict probabilities given features using the best pipeline.

            evaluate(X, y_true, threshold=0.5):
                Evaluate the performance of the best pipeline on the test set using various metrics.

            plot_roc_curve(X, y_true):
                Plot the Receiver Operating Characteristic (ROC) curve.

            plot_pr_curve(X, y_true):
                Plot the Precision-Recall curve.

            plot_confusion_matrix(X, y_true, threshold=0.5):
                Plot the confusion matrix.

            plot_optimization_history():
                Plot the optimization history showing the performance of trials over iterations.

            plot_param_importances():
                Plot the parameter importances obtained during optimization.

            plot_feature_importances(X):
                Plot the feature importances using SHAP values for the best pipeline.

            best_pipeline():
                Get the best machine learning pipeline obtained from optimization.

            best_score():
                Get the best CV score achieved during optimization.
        """

    def __init__(self):
        self._study = None
        self._best_pipeline = None
        self._best_score = None

    def optimize(self, X: pd.DataFrame, y: np.ndarray, metric: str='accuracy', select: bool=False,
     study_name: str='optimization', cv: Union[int, Iterable]=5, n_trials: int=100) -> None:
        """
        Optimize the machine learning pipeline using Optuna Bayesian optimization.

        Parameters:
            X (pd.DataFrame): Input features.
            y (np.ndarray): Target variable.
            metric (str): Metric to optimize (default is 'accuracy').
            select (bool): Whether to include feature selection in the pipeline (default is False).
            study_name (str): Name of the optimization study (default is 'optimization').
            cv (Union[int, Iterable]): Cross-validation strategy, can be an integer or a CV splitter (default is 5).
            n_trials (int): Number of optimization trials (default is 100).
        """
        numerical_features = [
            *X.select_dtypes(exclude=['object', 'category']).columns
        ]
        categorical_features = [
            *X.select_dtypes(include=['object', 'object']).columns
        ]

        def instantiate_pipeline(trial: Trial, numerical_features: List[str],
         categorical_features: List[str], select: bool=False) -> Pipeline:
            processor = _instantiate_processor(
                trial, numerical_features, categorical_features, select
            )
            estimator = _instantiate_estimator(trial)
            pipeline = Pipeline([
                ('processor', processor),
                ('estimator', estimator)
            ])
            return pipeline

        def objective(trial) -> float:
            pipeline = instantiate_pipeline(trial, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, scoring=metric, cv=cv)
            return np.min([np.mean(scores), np.median([scores])])

        op.logging.set_verbosity(op.logging.WARNING)
        self._study = op.create_study(study_name=study_name, direction='maximize')
        self._study.optimize(lambda trial: objective(trial), n_trials=n_trials)
        best_pipeline = instantiate_pipeline(self._study.best_trial, numerical_features, categorical_features, select)
        self._best_pipeline = best_pipeline.fit(X, y)
        self._best_score = self._study.best_value

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities given features using the best pipeline.

        Parameters:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        pred_prob = self._best_pipeline.predict_proba(X)[:, 1]
        return pred_prob

    def evaluate(self, X: pd.DataFrame, y_true: np.array, threshold: float=0.5) -> pd.DataFrame:
        """
        Evaluate the performance of the best pipeline on the test set using various metrics.

        Parameters:
            X (pd.DataFrame): Input features.
            y_true (np.array): True labels.
            threshold (float): Threshold for binary classification (default is 0.5).

        Returns:
            pd.DataFrame: DataFrame containing evaluation metrics.
        """
        pred_prob = self.predict(X)
        scores = {
            'AUC-ROC': [roc_auc_score(y_true, pred_prob)],
            'Accuracy': [accuracy_score(y_true, pred_prob >= threshold)],
            'Average Precision': [average_precision_score(y_true, pred_prob)],
            'Recall': [recall_score(y_true, pred_prob >= threshold)],
            'Precision': [precision_score(y_true, pred_prob >= threshold)],
            'F1': [f1_score(y_true, pred_prob >= threshold)]
        }
        return pd.DataFrame(scores).round(3)

    def plot_roc_curve(self, X: pd.DataFrame, y_true: np.array) -> None:
        """
        Plot the Receiver Operating Characteristic (ROC) curve.

        Parameters:
            X (pd.DataFrame): Input features.
            y_true (np.array): True labels.
        """
        pred_prob = self.predict(X)
        fpr, tpr, _ = roc_curve(y_true, pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    def plot_pr_curve(self, X: pd.DataFrame, y_true: np.array) -> None:
        """
        Plot the Precision-Recall curve.

        Parameters:
            X (pd.DataFrame): Input features.
            y_true (np.array): True labels.
        """
        pred_prob = self.predict(X)
        precision, recall, _ = precision_recall_curve(y_true, pred_prob)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'AUC = {pr_auc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='upper right')
        plt.show()

    def plot_confusion_matrix(self, X: pd.DataFrame, y_true: np.array, threshold: float=0.5) -> None:
        """
        Plot the confusion matrix.

        Parameters:
            X (pd.DataFrame): Input features.
            y_true (np.array): True labels.
            threshold (float): Threshold for binary classification (default is 0.5).
        """
        pred_prob = self.predict(X)
        y_pred = pred_prob >= threshold
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.title(f'Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()

    def plot_optimization_history(self) -> None:
        """
        Plot the optimization history showing the performance over trials.
        """
        fig = op.visualization.plot_optimization_history(self._study)
        fig.show()

    def plot_param_importances(self) -> None:
        """
        Plot the parameter importances obtained during optimization.
        """
        fig = op.visualization.plot_param_importances(self._study)
        fig.show()

    def plot_feature_importances(self, X: pd.DataFrame) -> None:
        """
        Plot the feature importances using SHAP values.

        Parameters:
            X (pd.DataFrame): Input features.
        """
        explainer = sp.Explainer(self._best_pipeline[-1])
        X_transformed = self._best_pipeline[0].transform(X)
        feature_names = [feature.split('__')[1] for feature in self._best_pipeline[0].get_feature_names_out()]
        shap_values = explainer.shap_values(X_transformed)
        sp.summary_plot(shap_values, X_transformed, feature_names)
    
    @property
    def best_pipeline(self) -> Pipeline:
        """
        Get the best machine learning pipeline obtained from optimization.

        Returns:
            Pipeline: Best machine learning pipeline.
        """
        return self._best_pipeline

    @property
    def best_score(self) -> float:
        """
        Get the best CV score achieved during optimization.

        Returns:
            float: Best score.
        """
        return self._best_score