from typing import Union

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from . import MLFLOW_ON

RANDOM_STATE = 42

# toggle MLflow autologging based on the environment variable
if MLFLOW_ON:
    mlflow.autolog()


def build_coefficients_dataframe(
    coefs: npt.ArrayLike,
    feature_labels: npt.ArrayLike,
) -> pd.DataFrame:
    """
    Build a dataframe with coefficients and feature labels.

    Parameters
    ----------
    coefs : npt.ArrayLike
        Coefficients.
    feature_labels : npt.ArrayLike
        Feature labels.

    Returns
    -------
    pd.DataFrame
        Dataframe with coefficients and feature labels.
    """
    return pd.DataFrame(
        data=coefs, index=feature_labels, columns=["coefficient"]
    ).sort_values(by="coefficient")


def build_classification_model_pipeline(
    classifier: sklearn.base.ClassifierMixin,
    preprocessor: sklearn.base.TransformerMixin = None,
) -> sklearn.pipeline.Pipeline:
    """
    Build a classification model pipeline.

    Parameters
    ----------
    classifier : sklearn.base.ClassifierMixin
        Classifier.
    preprocessor : sklearn.base.TransformerMixin, optional
        Preprocessor, by default None.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Model pipeline.
    """
    if preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("clf", classifier)])
    else:
        pipeline = Pipeline([("clf", classifier)])

    model = pipeline

    return model


def train_and_validate_classification_model(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    cv: sklearn.model_selection._split.BaseCrossValidator,
    classifier: sklearn.base.ClassifierMixin,
    preprocessor: sklearn.base.TransformerMixin = None,
):
    """
    Train and validate a classification model using cross-validation.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : Union[pd.Series, np.ndarray]
        Target.
    cv : sklearn.model_selection._split.BaseCrossValidator
        Cross-validation strategy.
    classifier : sklearn.base.ClassifierMixin
        Classifier.
    preprocessor : sklearn.base.TransformerMixin, optional
        Preprocessor, by default None.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with the scores.
    """

    model = build_classification_model_pipeline(
        classifier,
        preprocessor,
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ],
    )

    return scores


def grid_search_cv_classifier(
    classifier: sklearn.base.ClassifierMixin,
    param_grid: dict[str, npt.ArrayLike],
    cv: sklearn.model_selection._split.BaseCrossValidator,
    preprocessor: sklearn.base.TransformerMixin = None,
    return_train_score: bool = False,
    refit_metric: str = "roc_auc",
) -> GridSearchCV:
    """
    Perform a grid search cross-validation for a classifier.

    Parameters
    ----------
    classifier : sklearn.base.ClassifierMixin
        Classifier.
    param_grid : dict[str, npt.ArrayLike]
        Grid of hyperparameters.
    cv : sklearn.model_selection._split.BaseCrossValidator
        Cross-validation strategy.
    preprocessor : sklearn.base.TransformerMixin, optional
        Preprocessor, by default None.
    return_train_score : bool, optional
        Return train score, by default False.
    refit_metric : str, optional
        Metric to use for refitting, by default "roc_auc".

    Returns
    -------
    GridSearchCV
        Grid search object.
    """
    model = build_classification_model_pipeline(classifier, preprocessor)

    grid_search = GridSearchCV(
        model,
        cv=cv,
        param_grid=param_grid,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ],
        refit=refit_metric,
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    return grid_search


def summarize_model_performance(
    metrics_collection: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """
    Summarize model performance metrics in a DataFrame.

    Parameters
    ----------
    metrics_collection : dict[str, dict[str, float]]
        Dictionary with model names as keys and dictionaries with metrics as values

    Returns
    -------
    pd.DataFrame
        DataFrame with model performance metrics
    """

    for model_name, metric_values in metrics_collection.items():
        metrics_collection[model_name]["time_seconds"] = (
            metrics_collection[model_name]["fit_time"]
            + metrics_collection[model_name]["score_time"]
        )

    model_metrics_dataframe = (
        pd.DataFrame(metrics_collection)
        .T.reset_index()
        .rename(columns={"index": "model"})
    )

    model_metrics_dataframe_explode = model_metrics_dataframe.explode(
        model_metrics_dataframe.columns[1:].to_list()
    ).reset_index(drop=True)

    try:
        model_metrics_dataframe_explode = model_metrics_dataframe_explode.apply(
            pd.to_numeric
        )
    except ValueError:
        pass

    return model_metrics_dataframe_explode


def get_mean_and_std_from_grid_search_best_estimator(
    grid_search: GridSearchCV,
) -> pd.DataFrame:
    """
    Get mean and std scores for the best estimator from a GridSearchCV object.

    Parameters
    ----------
    grid_search : GridSearchCV
        Fitted GridSearchCV object.

    Returns
    -------
    pd.DataFrame
        DataFrame with mean and std scores for the best estimator
    """
    best_estimator_index = grid_search.best_index_
    cv_results = grid_search.cv_results_

    mean_scores = {}
    std_scores = {}

    for metric in grid_search.scoring:
        mean_scores[metric] = cv_results[f"mean_test_{metric}"][best_estimator_index]
        std_scores[metric] = cv_results[f"std_test_{metric}"][best_estimator_index]

    # create dataframe with mean and std scores for each metric
    # metric | mean | std
    df_mean_scores = pd.DataFrame(mean_scores, index=["score"]).T
    df_std_scores = pd.DataFrame(std_scores, index=["std"]).T
    df_mean_std_scores = pd.concat([df_mean_scores, df_std_scores], axis=1)

    return df_mean_std_scores
