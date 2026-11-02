"""Core data types, protocols, and result dataclasses."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class DownloadedAsset:
    """One locally downloaded STAC asset and its source identifiers.

    Attributes
    ----------
    path : pathlib.Path
        Local path to the downloaded raster asset.
    collection : str
        Source STAC collection.
    item_id : str
        Source STAC item identifier.
    asset_key : str
        Asset key used by the source collection, such as ``"B04"`` or
        ``"vv"``.
    """

    path: Path
    collection: str
    item_id: str
    asset_key: str


@runtime_checkable
class ClassifierModel(Protocol):
    """Protocol satisfied by fitted classifiers presenting a scikit-learn API."""

    def predict(self, X: Any) -> Any:
        ...


@runtime_checkable
class ClassifierModelWithProba(Protocol):
    """Protocol for classifiers supporting probability predictions."""

    def predict(self, X: Any) -> Any:
        ...

    def predict_proba(self, X: Any) -> Any:
        ...


@runtime_checkable
class SelectorProtocol(Protocol):
    """Protocol satisfied by feature selectors presenting transform."""

    def transform(self, X: Any) -> Any:
        ...
