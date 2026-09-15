"""Recursive feature elimination for raster-derived training samples."""

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV


@dataclass
class FeatureSelectionResult:
	"""Fitted feature selector and the data selected by it."""
	selector: RFECV
	X_train: Any
	X_test: Any
	selected_indices: np.ndarray
	selected_features: list[Any]

	@property
	def n_features(self) -> int:
		"""Return the number of features retained by RFECV."""
		return int(self.selector.n_features_)


def select_features(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray | None = None,
	*,
	estimator: BaseEstimator | None = None,
	feature_names: Sequence[Any] | None = None,
	cv: int = 5,
	scoring: str = "balanced_accuracy",
	step: int | float = 1,
	min_features_to_select: int = 1,
	n_jobs: int | None = -1,
) -> FeatureSelectionResult:
	"""Select an optimum feature subset using recursive feature elimination.

	RFECV is fitted only with ``X_train`` and ``y_train``. If supplied,
	``X_test`` is transformed after fitting and is never used during feature
	selection. The fitted selector should be retained for raster prediction.
	"""
	if X_train.ndim != 2 or y_train.ndim != 1:
		raise ValueError("X_train must be 2-D and y_train must be 1-D")
	if len(X_train) != len(y_train):
		raise ValueError("X_train and y_train must contain the same number of samples")
	if X_train.shape[1] == 0:
		raise ValueError("X_train must contain at least one feature")
	if X_test is not None:
		if X_test.ndim != 2:
			raise ValueError("X_test must be 2-D")
		if X_test.shape[1] != X_train.shape[1]:
			raise ValueError("X_test must contain the same features as X_train")
	if feature_names is not None and len(feature_names) != X_train.shape[1]:
		raise ValueError("feature_names must have one value per input feature")

	if estimator is None:
		estimator = RandomForestClassifier(
			n_estimators=200,
			random_state=42,
			n_jobs=n_jobs,
		)

	selector = RFECV(
		estimator=estimator,
		step=step,
		min_features_to_select=min_features_to_select,
		cv=cv,
		scoring=scoring,
		n_jobs=n_jobs,
	)
	selector.fit(X_train, y_train)

	selected_indices = np.flatnonzero(selector.support_)
	selected_features = (
		[feature_names[int(index)] for index in selected_indices]
		if feature_names is not None
		else selected_indices.tolist()
	)

	return FeatureSelectionResult(
		selector=selector,
		X_train=selector.transform(X_train),
		X_test=selector.transform(X_test) if X_test is not None else None,
		selected_indices=selected_indices,
		selected_features=selected_features,
	)
