"""Hyperparameter tuning and held-out evaluation for classifiers."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
)
from sklearn.model_selection import RandomizedSearchCV


@dataclass
class ModelTuningResult:
	"""The fitted result and diagnostics from randomized hyperparameter search."""

	model: BaseEstimator
	best_params: dict[str, Any]
	best_score: float
	search: RandomizedSearchCV


@dataclass
class ModelEvaluation:
	"""Metrics calculated on a held-out test set."""

	metrics: dict[str, float]
	report: dict[str, Any]
	confusion_matrix: np.ndarray


def tune_model(
	estimator: BaseEstimator,
	param_distributions: Mapping[str, Any],
	X_train: np.ndarray,
	y_train: np.ndarray,
	*,
	n_iter: int = 20,
	cv: int | Any = 5,
	scoring: str = "balanced_accuracy",
	n_jobs: int | None = -1,
	random_state: int = 42,
	verbose: int = 0,
	return_train_score: bool = False,
) -> ModelTuningResult:
	"""Tune any scikit-learn-compatible classifier with randomized search.

	``estimator`` may be a Random Forest, XGBoost, LightGBM, or another
	classifier implementing the scikit-learn estimator interface. The caller
	supplies parameter names and candidate values or scipy distributions.
	The returned model is fitted on all training samples using the best
	parameters; the test set must be evaluated separately with
	:func:`evaluate_model`.
	"""
	if X_train.ndim != 2 or y_train.ndim != 1:
		raise ValueError("X_train must be 2-D and y_train must be 1-D")
	if len(X_train) != len(y_train):
		raise ValueError("X_train and y_train must contain the same number of samples")
	if X_train.shape[1] == 0:
		raise ValueError("X_train must contain at least one feature")
	if n_iter < 1:
		raise ValueError("n_iter must be at least 1")

	search = RandomizedSearchCV(
		estimator=estimator,
		param_distributions=dict(param_distributions),
		n_iter=n_iter,
		cv=cv,
		scoring=scoring,
		n_jobs=n_jobs,
		random_state=random_state,
		verbose=verbose,
		return_train_score=return_train_score,
		refit=True,
	)
	search.fit(X_train, y_train)

	return ModelTuningResult(
		model=search.best_estimator_,
		best_params=search.best_params_,
		best_score=float(search.best_score_),
		search=search,
	)


def evaluate_model(
	model: BaseEstimator,
	X_test: np.ndarray,
	y_test: np.ndarray,
	*,
	labels: np.ndarray | None = None,
) -> ModelEvaluation:
	"""Evaluate a fitted classifier on held-out samples."""
	if X_test.ndim != 2 or y_test.ndim != 1:
		raise ValueError("X_test must be 2-D and y_test must be 1-D")
	if len(X_test) != len(y_test):
		raise ValueError("X_test and y_test must contain the same number of samples")

	predictions = model.predict(X_test)
	metrics = {
		"accuracy": float(accuracy_score(y_test, predictions)),
		"balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
		"f1_macro": float(f1_score(y_test, predictions, average="macro")),
		"f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
	}
	report = classification_report(
		y_test,
		predictions,
		labels=labels,
		output_dict=True,
		zero_division=0,
	)
	matrix = confusion_matrix(y_test, predictions, labels=labels)
	return ModelEvaluation(
		metrics=metrics,
		report=report,
		confusion_matrix=matrix,
	)
