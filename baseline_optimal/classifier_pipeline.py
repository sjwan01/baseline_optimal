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
    """A class for optimizing and evaluating a machine learning classifier using Optuna.

    Attributes:
        _study (optuna.study.Study): Optuna study for hyperparameter optimization.
        _best_pipeline (Pipeline): Best scikit-learn pipeline after optimization.
        _best_score (float): Best performance metric score after optimization.
        _pred_prob (np.ndarray): Predicted probabilities after model fitting.

    Methods:
        optimize(X, y, metric='accuracy', select=False, study_name='optimization', cv=5, n_trials=100):
            Optimize the classifier by tuning hyperparameters using Optuna.

        fit(X, y):
            Fit the classifier with the best hyperparameters on the provided data.

        predict_proba(X, y, threshold=0.5) -> Tuple[np.ndarray, pd.DataFrame]:
            Evaluate the classifier on new data and return predicted probabilities and evaluation metrics.

        plot_roc_curve(y_true):
            Plot the Receiver Operating Characteristic (ROC) curve.

        plot_pr_curve(y_true):
            Plot the Precision-Recall (PR) curve.

        plot_confusion_matrix(y_true, threshold):
            Plot the confusion matrix.

        plot_optimization_history():
            Plot the optimization history during the hyperparameter search.

        plot_param_importances():
            Plot the importances of hyperparameters based on their impact on the objective function.

        plot_feature_importances(X):
            Plot the SHAP (SHapley Additive exPlanations) summary plot for feature importances.

    Properties:
        best_pipeline: Get the best pipeline discovered during optimization.
        best_score: Get the best score achieved during optimization.
    """

    def __init__(self):
        """
        Initialize an instance of ClassTask.

        Attributes:
        - _study (optuna.study.Study): Optuna study for hyperparameter optimization.
        - _best_pipeline (Pipeline): Best scikit-learn pipeline after optimization.
        - _best_score (float): Best performance metric score after optimization.
        - _pred_prob (np.ndarray): Predicted probabilities after model fitting.
        """
        self._study = None
        self._best_pipeline = None
        self._best_score = None
        self._pred_prob = None

    def optimize(self, X: pd.DataFrame, y: np.ndarray, metric: str='accuracy', select: bool=False,
     study_name: str='optimization', cv: Union[int, Iterable]=5, n_trials: int=100) -> None:
        """
        Optimize the classifier hyperparameters using Optuna.

        Args:
        - X (pd.DataFrame): Features for training.
        - y (np.ndarray): Target labels for training.
        - metric (str): Performance metric for optimization.
        - select (bool): Whether to perform feature selection.
        - study_name (str): Name of the Optuna study.
        - cv (Union[int, Iterable]): Number of cross-validation folds or an iterable for custom cross-validation.
        - n_trials (int): Number of optimization trials.

        Returns:
        - None
        """
        numerical_features = [
            *X.select_dtypes(exclude=['object', 'category']).columns
        ]
        categorical_features = [
            *X.select_dtypes(include=['object', 'object']).columns
        ]

        def instantiate_pipeline(trial: Trial, numerical_features: List[str],
         categorical_features: List[str], select: bool=False) -> Pipeline:
            """
            Instantiate and return a scikit-learn pipeline.

            Args:
            - trial (optuna.trial.Trial): Optuna trial object.
            - numerical_features (List[str]): List of numerical features.
            - categorical_features (List[str]): List of categorical features.
            - select (bool): Whether to perform feature selection.

            Returns:
            - Pipeline: Scikit-learn pipeline.
            """
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
            """
            Objective function for hyperparameter optimization.

            Args:
            - trial: Optuna trial object.

            Returns:
            - float: Performance metric score.
            """
            pipeline = instantiate_pipeline(trial, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, scoring=metric, cv=cv)
            return np.min([np.mean(scores), np.median([scores])])

        op.logging.set_verbosity(op.logging.WARNING)
        self._study = op.create_study(study_name=study_name, direction='maximize')
        self._study.optimize(lambda trial: objective(trial), n_trials=n_trials)
        self._best_pipeline = instantiate_pipeline(self._study.best_trial, numerical_features, categorical_features, select)
        self._best_score = self._study.best_value

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Fit the optimal pipeline on the training data.

        Args:
        - X (pd.DataFrame): Features for training.
        - y (np.ndarray): Target labels for training.

        Returns:
        - None
        """
        self._best_pipeline.fit(X, y)

    def predict_proba(self, X: pd.DataFrame, y_true: np.array, threshold: float=0.5) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Predict probabilities on the input data and calculate evaluation metrics.

        Args:
        - X (pd.DataFrame): Features for prediction.
        - y_true (np.array): True labels for evaluation.
        - threshold (float): Classification threshold for binary predictions.

        Returns:
        - Tuple[np.ndarray, pd.DataFrame]: Predicted probabilities and a DataFrame of evaluation metrics.
        """
        self._pred_prob = self._best_pipeline.predict_proba(X)[:, 1]
        scores = {
            'AUC-ROC': [roc_auc_score(y_true, self._pred_prob)],
            'Accuracy': [accuracy_score(y_true, self._pred_prob >= threshold)],
            'Average Precision': [average_precision_score(y_true, self._pred_prob)],
            'Recall': [recall_score(y_true, self._pred_prob >= threshold)],
            'Precision': [precision_score(y_true, self._pred_prob >= threshold)],
            'F1': [f1_score(y_true, self._pred_prob >= threshold)]
        }
        return self._pred_prob, pd.DataFrame(scores).round(3)

    def plot_roc_curve(self, y_true: np.array) -> None:
        """
        Plot the ROC curve.

        Args:
        - y_true (np.array): True labels for evaluation.

        Returns:
        - None
        """
        fpr, tpr, _ = roc_curve(y_true, self._pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    def plot_pr_curve(self, y_true: np.array) -> None:
        """
        Plot the Precision-Recall curve.

        Args:
        - y_true (np.array): True labels for evaluation.

        Returns:
        - None
        """
        precision, recall, _ = precision_recall_curve(y_true, self._pred_prob)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'AUC = {pr_auc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='upper right')
        plt.show()

    def plot_confusion_matrix(self, y_true: np.array, threshold: float=0.5) -> None:
        """
        Plot the confusion matrix.

        Args:
        - y_true (np.array): True labels for evaluation.
        - threshold (float): Classification threshold for binary predictions.

        Returns:
        - None
        """
        y_pred = self._pred_prob >= threshold
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.title(f'Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()

    def plot_optimization_history(self) -> None:
        """
        Plot the optimization history during the hyperparameter search.

        Returns:
        - None
        """
        fig = op.visualization.plot_optimization_history(self._study)
        fig.show()

    def plot_param_importances(self) -> None:
        """
        Plot the parameter importances during the hyperparameter search.

        Returns:
        - None
        """
        fig = op.visualization.plot_param_importances(self._study)
        fig.show()

    def plot_feature_importances(self, X: pd.DataFrame) -> None:
        """
        Plot the feature importances using SHAP values.

        Args:
        - X (pd.DataFrame): Features for evaluation.

        Returns:
        - None
        """
        explainer = sp.Explainer(self._best_pipeline[-1])
        X_transformed = self._best_pipeline[0].transform(X)
        feature_names = [feature.split('__')[1] for feature in self._best_pipeline[0].get_feature_names_out()]
        shap_values = explainer.shap_values(X_transformed)
        sp.summary_plot(shap_values, X_transformed, feature_names)
    
    @property
    def best_pipeline(self) -> Pipeline:
        """
        Get the best scikit-learn pipeline.

        Returns:
        - Pipeline: Best scikit-learn pipeline.
        """
        return self._best_pipeline

    @property
    def best_score(self) -> float:
        """
        Get the best performance metric score.

        Returns:
        - float: Best performance metric score.
        """
        return self._best_score