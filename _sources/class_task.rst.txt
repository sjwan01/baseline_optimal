baseline\_optimal.class\_task
==============================

The module currently supports classification tasks given balanced dataset.

Usage
------

To prepare your data, make sure to:

1. Remove features that machine learning models can't process, such as names and zip codes.
2. Split the training and the test data, and encode the target variable if necessary,

To start with, declare a ``ClassTask`` object:

.. code-block:: python

    from baseline_optimal import ClassTask

    task = ClassTask()

Search through the parameter space and optimize for the accuracy score by default. Other supported metrics for classification tasks can be found through `sklearn <https://scikit-learn.org/stable/modules/model_evaluation.html>`_. 

Set ``select=True`` when the dataset is large to enable feature selection prior to fitting models. This can speed up the optimization process by using a subset of the feature space, but it may lead to suboptimal results given relatively smaller dataset.

``cv`` defines the number of folds within each iteration, and it can also take a `CV splitter <https://scikit-learn.org/stable/modules/cross_validation.html>`_. The ``optimize`` function aims to discover the optimal machine learning pipeline, incorporating the best data processor and estimator with optimized hyperparameters. Subsequently, it fits this optimal pipeline to the training data.

.. code-block:: python

    task.optimize(X=X_train, y=y_train, metric='accuracy', select=False,
                  study_name='optimization', cv=5, n_trials=100)

Access the best CV score.

.. code-block:: python

    print(task.best_score)

After fitting the data, one option is to use the ``ClassTask`` object for probability estimates and evaluation on the test set. You may customize the classification threshold.

.. code-block:: python

    pred_prob = study.predict(X=X_test)

    scores = study.evaluate(X=X_test, y_true=y_test, threshold=0.5)

Or, you may obtain the optimal ``imblearn.pipeline.Pipeline`` object. Check here_.

.. _here: https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html

.. code-block:: python

    best_pipeline = study.best_pipeline

    pred_prob = best_pipeline.predict_proba(X_test)

    best_processor = best_pipeline.named_steps['processor']
    best_estimator = best_pipeline.named_steps['estimator']

Obtain the ``imblearn.pipeline.Pipeline`` object will allow you to do much more. The ``imblearn.pipeline.Pipeline`` object is returned here to allow resampling for imbalanced data during optimization in future versions. It shares similar API with ``sklearn.pipeline.Pipeline``.

Documentation
----------------

.. automodule:: baseline_optimal.class_task
   :members:
   :undoc-members:
   :show-inheritance: