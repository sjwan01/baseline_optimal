import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna as op
import shap as sp
from optuna import Trial
from typing import List, Union, Iterable
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from ._processor import _instantiate_processor
from ._classifier import _instantiate_estimator


class ClassTask:
    """
        A `ClassTask` object manages the optimization process given classification tasks and provides methods
        to evaluate performance of the optimal machine learning pipeline on unforeseen data.
    """

    def __init__(self):
        self._study = None
        self._best_pipeline = None
        self._best_score = None

    def optimize(self, X: pd.DataFrame, y: np.ndarray, metric: str='accuracy', select: bool=False,
     study_name: str='optimization', cv: Union[int, Iterable]=5, n_trials: int=100) -> None:
        """
        Optimize the machine learning pipeline by tuning its components and estimator hyperparameters through, 
        by default, 100 trials based on the specified evaluation metric using Bayesian optimization with the 
        Optuna library. Use 5-fold cross validation by default.

        Parameters:
            X (pd.DataFrame): Training features.
            y (np.ndarray): Training labels.
            metric (str): Classification metric.
            select (bool): Whether to perform feature selection prior to fitting models. Set `True` to save computing resources.
            study_name (str): Name of the optimization study.
            cv (Union[int, Iterable]): Cross-validation strategy.
            n_trials (int): Number of optimization trials.
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
        Probability estimates of the new data using the optimal pipeline.

        Parameters:
            X (pd.DataFrame): Test features.

        Returns:
            np.ndarray: Probability estimates.
        """
        pred_prob = self._best_pipeline.predict_proba(X)[:, 1]
        return pred_prob

    def evaluate(self, X: pd.DataFrame, y_true: np.array, threshold: float=0.5) -> pd.DataFrame:
        """
        Evaluate performance of the optimal pipeline on the test data using threshold-based and ranking-based metrics.
        The classification threshold is set to be 0.5 by default.

        Parameters:
            X (pd.DataFrame): Test features.
            y_true (np.array): Test labels.
            threshold (float): Binary classification threshold.

        Returns:
            pd.DataFrame: A pd.DataFrame object containing evaluation results.
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
            X (pd.DataFrame): Features.
            y_true (np.array): Labels.
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
            X (pd.DataFrame): Features.
            y_true (np.array): Labels.
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
            X (pd.DataFrame): Features.
            y_true (np.array): Labels.
            threshold (float): Binary classification threshold.
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
        Plot the parameter importances during optimization.
        """
        fig = op.visualization.plot_param_importances(self._study)
        fig.show()

    def plot_feature_importances(self, X: pd.DataFrame) -> None:
        """
        Plot the feature importances based on SHAP values.

        Parameters:
            X (pd.DataFrame): Features.
        """
        explainer = sp.Explainer(self._best_pipeline[-1])
        X_transformed = self._best_pipeline[0].transform(X)
        feature_names = [feature.split('__')[1] for feature in self._best_pipeline[0].get_feature_names_out()]
        shap_values = explainer.shap_values(X_transformed)
        sp.summary_plot(shap_values, X_transformed, feature_names)
    
    @property
    def best_pipeline(self) -> Pipeline:
        """
        Get the optimal machine learning pipeline obtained from optimization.

        Returns:
            imblearn.pipeline.Pipeline: The optimal machine learning pipeline.
        """
        return self._best_pipeline

    @property
    def best_score(self) -> float:
        """
        Get the best CV score achieved during optimization.

        Returns:
            float: The best CV score.
        """
        return self._best_score