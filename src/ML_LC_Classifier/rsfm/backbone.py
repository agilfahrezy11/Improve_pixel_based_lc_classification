"""Backbone protocol and OpenGeoAI loader for Remote Sensing Foundation Models."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, Sequence, runtime_checkable
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter


Float32Array = NDArray[np.float32]


def _import_geoai() -> Any:
    """Import ``geoai`` lazily with actionable installation instructions."""
    try:
        module = importlib.import_module("geoai")
    except ImportError as exc:
        raise ImportError(
            "OpenGeoAI support requires the optional 'rsfm' dependencies. "
            'Install them with: pip install -e ".[rsfm]"'
        ) from exc
    if module is None:
        raise ImportError(
            "OpenGeoAI support requires the optional 'rsfm' dependencies. "
            'Install them with: pip install -e ".[rsfm]"'
        )
    return module


@runtime_checkable
class RSFMBackbone(Protocol):
    """Protocol satisfied by pluggable RSFM feature-extraction backbones."""

    @property
    def embed_dim(self) -> int:
        """Number of embedding channels produced per pixel."""
        ...

    def extract_features(
        self,
        chips: NDArray[np.floating[Any]],
        *,
        wavelengths_um: Sequence[float] | None = None,
    ) -> Float32Array:
        """Map ``(batch, channels, height, width)`` chips to ``(batch, embed_dim, height, width)``."""
        ...


class OpenGeoAIBackboneAdapter:
    """Adapter wrapping a backbone loaded from ``geoai.load_foundation_model``."""

    def __init__(self, model: Any, model_name: str, embed_dim: int | None = None) -> None:
        self.model = model
        self.model_name = model_name
        inferred_dim = embed_dim if embed_dim is not None else getattr(model, "embed_dim", None)
        if inferred_dim is None:
            inferred_dim = getattr(model, "num_features", 64)
        self._embed_dim = int(inferred_dim)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def extract_features(
        self,
        chips: NDArray[np.floating[Any]],
        *,
        wavelengths_um: Sequence[float] | None = None,
    ) -> Float32Array:
        """Extract dense spatial embeddings using the wrapped OpenGeoAI model."""
        torch = importlib.import_module("torch")
        tensor = torch.as_tensor(np.asarray(chips, dtype=np.float32))
        kwargs: dict[str, Any] = {}
        if wavelengths_um is not None:
            kwargs["wavelengths"] = list(wavelengths_um)
        with torch.no_grad():
            if hasattr(self.model, "forward_features"):
                out = self.model.forward_features(tensor, **kwargs)
            else:
                out = self.model(tensor, **kwargs)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        arr = out.detach().cpu().numpy().astype(np.float32, copy=False)
        return arr


def list_opengeoai_models(**kwargs: Any) -> Any:
    """List available foundation models from the OpenGeoAI catalog."""
    geoai = _import_geoai()
    return geoai.list_foundation_models(**kwargs)


def load_opengeoai_backbone(
    model_name: str,
    *,
    embed_dim: int | None = None,
    **kwargs: Any,
) -> OpenGeoAIBackboneAdapter:
    """Load a foundation model backbone via ``geoai.load_foundation_model``."""
    geoai = _import_geoai()
    model = geoai.load_foundation_model(model_name, **kwargs)
    return OpenGeoAIBackboneAdapter(model=model, model_name=model_name, embed_dim=embed_dim)


class SimpleCPUBackbone:
    """Deterministic lightweight spatial-spectral backbone for CPU testing and prototyping."""

    def __init__(self, embed_dim: int = 16, seed: int = 42) -> None:
        if embed_dim < 1:
            raise ValueError("embed_dim must be at least 1")
        self._embed_dim = int(embed_dim)
        self.seed = int(seed)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def extract_features(
        self,
        chips: NDArray[np.floating[Any]],
        *,
        wavelengths_um: Sequence[float] | None = None,
    ) -> Float32Array:
        """Compute deterministic spatial-spectral embeddings for ``(B, C, H, W)`` chips."""
        arr = np.asarray(chips, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError("chips must have shape (batch, channels, height, width)")
        batch, channels, height, width = arr.shape
        if wavelengths_um is not None and len(wavelengths_um) != channels:
            raise ValueError(
                f"wavelengths_um length ({len(wavelengths_um)}) must match channels ({channels})"
            )

        # Combine local pixel values with 3x3 neighborhood context
        context = uniform_filter(arr, size=(1, 1, 3, 3), mode="nearest")
        combined = np.concatenate([arr, context], axis=1)  # (B, 2*C, H, W)

        rng = np.random.default_rng(self.seed + channels)
        proj = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(2 * channels),
            size=(2 * channels, self._embed_dim),
        ).astype(np.float32)

        if wavelengths_um is not None:
            wl = np.asarray(wavelengths_um, dtype=np.float32)
            wl_scale = np.concatenate([wl, wl])[:, None]
            proj = proj * (1.0 + 0.1 * np.sin(wl_scale * np.pi))

        flat = np.moveaxis(combined, 1, -1).reshape(-1, 2 * channels)
        embedded = np.tanh(flat @ proj)
        return np.moveaxis(
            embedded.reshape(batch, height, width, self._embed_dim),
            -1,
            1,
        ).astype(np.float32, copy=False)
