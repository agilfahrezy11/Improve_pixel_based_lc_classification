"""
Model Hyperparameter Tuning and Evaluation

This module provides function to tune the hyperparameters of a classifier using grid or randomized search.
Additionally, it provides functionality to evaluate the performance of the fitter model. Supported Classifiers are as follows:
- Random Forest
- Extreme Gradient Boosting (XGBoost)
- Extremely Randomized Trees (ExtraTrees/ERT)
- Light Gradient Boosting Machine (LGBM)

More algorithms can be added by passing a scikit-learn compatible estimator to the ``tune_model`` function.
"""
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, cast
import numpy as np
from lightgbm import LGBMClassifier #LIGHT gradient boosting machine
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, RandomizedSearchCV

Classifier = str | BaseEstimator

class ClassifierModel(Protocol):
	"""Minimal fitted-classifier interface required for evaluation."""

	def predict(self, X: np.ndarray) -> np.ndarray:
		...

def build_classifier(
	classifier: str,
	*,
	n_jobs: int | None = -1,
	random_state: int = 42,
) -> BaseEstimator:
	"""Build one of the classifiers supported by the package."""
	name = classifier.lower().replace(" ", "").replace("-", "").replace("_", "")
	if name in {"randomforest", "rf"}:
		return RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
	if name == "extratrees":
		return ExtraTreesClassifier(random_state=random_state, n_jobs=n_jobs)
	if name in {"xgboost", "xgb"}:
		try:
			from xgboost import XGBClassifier
		except ImportError as error:
			raise ImportError(
				"XGBoost support requires the optional dependency: pip install "
				"ml-lc-classifier[boosting]"
			) from error
		return XGBClassifier(random_state=random_state, n_jobs=n_jobs)
	if name in {"lightgbm", "lgbm"}:
		return cast(
			BaseEstimator,
			LGBMClassifier(random_state=random_state, n_jobs=n_jobs, verbosity=-1),
		)
	raise ValueError(
		f"Unsupported classifier {classifier!r}. Choose RandomForest, ExtraTrees, "
		"XGBoost, LightGBM, or pass a scikit-learn estimator."
	)


def _make_search(
	search_method: str,
	estimator: BaseEstimator,
	parameter_space: Mapping[str, Any],
	*,
	n_iter: int,
	cv: int | BaseCrossValidator,
	scoring: str,
	n_jobs: int | None,
	random_state: int,
	verbose: int,
	return_train_score: bool,
) -> Any:
	common: dict[str, Any] = dict(
		estimator=estimator,
		cv=cv,
		scoring=scoring,
		n_jobs=n_jobs,
		verbose=verbose,
		return_train_score=return_train_score,
	)
	method = search_method.lower()
	if method == "grid":
		return GridSearchCV(**common, param_grid=dict(parameter_space))
	if method in {"random", "randomized"}:
		if n_iter < 1:
			raise ValueError("n_iter must be at least 1")
		return RandomizedSearchCV(
			**common,
			param_distributions=dict(parameter_space),
			n_iter=n_iter,
			random_state=random_state,
		)
	raise ValueError("search_method must be 'random' or 'grid'")

@dataclass
class ModelTuningResult:
	"""The fitted result and diagnostics from randomized hyperparameter search."""

	model: BaseEstimator
	best_params: dict[str, Any]
	best_score: float
	search: Any


@dataclass
class ModelEvaluation:
	"""Metrics calculated on a held-out test set."""

	metrics: dict[str, float]
	report: dict[str, Any]
	confusion_matrix: np.ndarray


def tune_model(
	estimator: Classifier | None = None,
	param_distributions: Mapping[str, Any] | None = None,
	X_train: np.ndarray | None = None,
	y_train: np.ndarray | None = None,
	*,
	classifier: str | None = None,
	parameter_space: Mapping[str, Any] | None = None,
	search_method: str = "random",
	n_iter: int = 20,
	cv: int | BaseCrossValidator = 5,
	scoring: str = "balanced_accuracy",
	n_jobs: int | None = -1,
	random_state: int = 42,
	verbose: int = 0,
	return_train_score: bool = False,
) -> ModelTuningResult:
	"""Tune a named or custom classifier with grid or randomized search.

	Use ``classifier="RandomForest"`` (or ``"ExtraTrees"``, ``"XGBoost"``,
	``"LightGBM"``) to let this package construct the estimator. Supply
	parameter names and candidate values through ``parameter_space``. The
	``param_distributions`` name remains a backwards-compatible alias, and
	passing an estimator directly is also supported.
	The returned model is fitted on all training samples using the best
	parameters; the test set must be evaluated separately with
	:func:`evaluate_model`.
	"""
	if X_train is None or y_train is None:
		raise ValueError("X_train and y_train are required")
	if X_train.ndim != 2 or y_train.ndim != 1:
		raise ValueError("X_train must be 2-D and y_train must be 1-D")
	if len(X_train) != len(y_train):
		raise ValueError("X_train and y_train must contain the same number of samples")
	if X_train.shape[1] == 0:
		raise ValueError("X_train must contain at least one feature")
	if classifier is not None and estimator is not None:
		raise ValueError("Provide either classifier or estimator, not both")
	if parameter_space is not None and param_distributions is not None:
		raise ValueError("Provide either parameter_space or param_distributions, not both")
	parameter_space = parameter_space or param_distributions
	if parameter_space is None:
		raise ValueError("parameter_space is required")
	if classifier is not None:
		estimator = build_classifier(
			classifier, n_jobs=n_jobs, random_state=random_state
		)
	elif estimator is None:
		raise ValueError("Provide classifier or estimator")
	elif isinstance(estimator, str):
		estimator = build_classifier(
			estimator, n_jobs=n_jobs, random_state=random_state
		)

	search = _make_search(
		search_method=search_method,
		estimator=estimator,
		parameter_space=parameter_space,
		n_iter=n_iter,
		cv=cv,
		scoring=scoring,
		n_jobs=n_jobs,
		random_state=random_state,
		verbose=verbose,
		return_train_score=return_train_score,
	)
	search.fit(X_train, y_train)

	return ModelTuningResult(
		model=search.best_estimator_,
		best_params=search.best_params_,
		best_score=float(search.best_score_),
		search=search,
	)


def evaluate_model(
	model: ClassifierModel,
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
	report = cast(
		dict[str, Any],
		classification_report(
			y_test,
			predictions,
			labels=labels,
			output_dict=True,
			zero_division=0,
		),
	)
	matrix = confusion_matrix(y_test, predictions, labels=labels)
	return ModelEvaluation(
		metrics=metrics,
		report=report,
		confusion_matrix=matrix,
	)
