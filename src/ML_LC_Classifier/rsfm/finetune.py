"""LoRA fine-tuning and portable checkpoint bundles for RSFM backbones."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from os import PathLike
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray

from .backbone import OpenGeoAIBackboneAdapter, RSFMBackbone
from .chips import RSFMChipDataset


PathLikeInput = str | PathLike[str]


def _softmax(logits: NDArray[np.float32], axis: int = -1) -> NDArray[np.float32]:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return (exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)).astype(np.float32, copy=False)


class RSFMCheckpointBundle:
    """Portable bundle containing LoRA adapter weights, head weights, and metadata."""

    def __init__(
        self,
        *,
        backbone: RSFMBackbone,
        class_values: tuple[int, ...],
        band_names: tuple[str, ...],
        wavelengths_um: tuple[float, ...] | None,
        mean: NDArray[np.float32],
        std: NDArray[np.float32],
        lora_rank: int,
        lora_alpha: float,
        lora_a: NDArray[np.float32],
        lora_b: NDArray[np.float32],
        head_weight: NDArray[np.float32],
        head_bias: NDArray[np.float32],
    ) -> None:
        self.backbone = backbone
        self.class_values = tuple(int(c) for c in class_values)
        self.band_names = tuple(band_names)
        self.wavelengths_um = tuple(float(w) for w in wavelengths_um) if wavelengths_um is not None else None
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_a = np.asarray(lora_a, dtype=np.float32)
        self.lora_b = np.asarray(lora_b, dtype=np.float32)
        self.head_weight = np.asarray(head_weight, dtype=np.float32)
        self.head_bias = np.asarray(head_bias, dtype=np.float32)

    @property
    def embed_dim(self) -> int:
        return self.backbone.embed_dim

    def _normalize(self, chips: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        arr = np.asarray(chips, dtype=np.float32)
        return (arr - self.mean[None, :, None, None]) / self.std[None, :, None, None]

    def predict_embeddings(self, chips: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        """Return LoRA-adapted embeddings of shape ``(B, D, H, W)``."""
        norm = self._normalize(chips)
        base_feats = self.backbone.extract_features(norm, wavelengths_um=self.wavelengths_um)
        scale = np.float32(self.lora_alpha / max(1, self.lora_rank))
        # Apply low-rank residual (B, C, H, W) -> (B, H, W, C) @ A @ B -> (B, D, H, W)
        x_hwc = np.moveaxis(norm, 1, -1)
        delta_hwc = scale * ((x_hwc @ self.lora_a) @ self.lora_b)
        delta = np.moveaxis(delta_hwc, -1, 1)
        return (base_feats + delta).astype(np.float32, copy=False)

    def predict_proba(self, chips: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        """Return per-pixel class probabilities of shape ``(B, K, H, W)``."""
        emb = self.predict_embeddings(chips)
        emb_hwc = np.moveaxis(emb, 1, -1)
        logits_hwc = emb_hwc @ self.head_weight + self.head_bias
        probs_hwc = _softmax(logits_hwc, axis=-1)
        return np.moveaxis(probs_hwc, -1, 1).astype(np.float32, copy=False)

    def save(self, checkpoint_dir: PathLikeInput) -> Path:
        """Save LoRA weights, head weights, and metadata manifest to ``checkpoint_dir``."""
        out_dir = Path(checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            out_dir / "weights.npz",
            mean=self.mean,
            std=self.std,
            lora_a=self.lora_a,
            lora_b=self.lora_b,
            head_weight=self.head_weight,
            head_bias=self.head_bias,
        )
        manifest = {
            "embed_dim": self.embed_dim,
            "class_values": list(self.class_values),
            "band_names": list(self.band_names),
            "wavelengths_um": list(self.wavelengths_um) if self.wavelengths_um is not None else None,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out_dir


@dataclass(frozen=True)
class RSFMFineTuneResult:
    """Result returned by :func:`finetune_rsfm`."""

    bundle: RSFMCheckpointBundle
    loss_history: tuple[float, ...]


def finetune_rsfm(
    dataset: RSFMChipDataset,
    *,
    backbone: RSFMBackbone,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    epochs: int = 25,
    lr: float = 0.05,
    ignore_index: int = 0,
    seed: int = 42,
) -> RSFMFineTuneResult:
    """Fine-tune an RSFM backbone with LoRA adapters and a linear head on sparse masks.

    Base backbone weights remain frozen. Only the low-rank adapter matrices
    and classification head parameters are updated using cross-entropy loss
    masked on ``masks != ignore_index``.
    """
    if lora_rank < 1:
        raise ValueError("lora_rank must be at least 1")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    if isinstance(backbone, OpenGeoAIBackboneAdapter):
        importlib.import_module("torch")
        importlib.import_module("peft")

    norm_chips = (
        dataset.chips - dataset.mean[None, :, None, None]
    ) / dataset.std[None, :, None, None]

    # Base backbone stays frozen; extract base representations once
    base_emb = backbone.extract_features(norm_chips, wavelengths_um=dataset.wavelengths_um)

    n_chips, n_channels, height, width = norm_chips.shape
    embed_dim = backbone.embed_dim
    num_classes = len(dataset.class_values)
    class_to_idx = {cls_id: idx for idx, cls_id in enumerate(dataset.class_values)}

    valid_mask = dataset.masks != ignore_index
    if not valid_mask.any():
        raise ValueError("No labeled pixels found outside ignore_index")

    x_flat = np.moveaxis(norm_chips, 1, -1)[valid_mask]  # (M, C)
    e0_flat = np.moveaxis(base_emb, 1, -1)[valid_mask]  # (M, D)
    y_raw = dataset.masks[valid_mask]
    y_idx = np.array([class_to_idx[int(v)] for v in y_raw], dtype=np.int64)
    n_valid = len(y_idx)

    y_onehot = np.zeros((n_valid, num_classes), dtype=np.float32)
    y_onehot[np.arange(n_valid), y_idx] = 1.0

    rng = np.random.default_rng(seed)
    lora_a = rng.normal(0.0, 0.05, size=(n_channels, lora_rank)).astype(np.float32)
    lora_b = np.zeros((lora_rank, embed_dim), dtype=np.float32)
    head_weight = rng.normal(0.0, 0.05, size=(embed_dim, num_classes)).astype(np.float32)
    head_bias = np.zeros((num_classes,), dtype=np.float32)

    scale = np.float32(lora_alpha / lora_rank)
    losses: list[float] = []

    for _ in range(epochs):
        h_lora = x_flat @ lora_a  # (M, r)
        emb = e0_flat + scale * (h_lora @ lora_b)  # (M, D)
        logits = emb @ head_weight + head_bias  # (M, K)
        probs = _softmax(logits, axis=1)

        loss = float(-np.mean(np.log(np.clip(probs[np.arange(n_valid), y_idx], 1e-7, 1.0))))
        losses.append(loss)

        d_logits = (probs - y_onehot) / np.float32(n_valid)
        grad_head_w = emb.T @ d_logits
        grad_head_b = np.sum(d_logits, axis=0)

        d_emb = d_logits @ head_weight.T  # (M, D)
        grad_b = scale * (h_lora.T @ d_emb)
        d_h_lora = scale * (d_emb @ lora_b.T)
        grad_a = x_flat.T @ d_h_lora

        head_weight -= np.float32(lr) * grad_head_w
        head_bias -= np.float32(lr) * grad_head_b
        lora_b -= np.float32(lr) * grad_b
        lora_a -= np.float32(lr) * grad_a

    bundle = RSFMCheckpointBundle(
        backbone=backbone,
        class_values=dataset.class_values,
        band_names=dataset.band_names,
        wavelengths_um=dataset.wavelengths_um,
        mean=dataset.mean,
        std=dataset.std,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_a=lora_a,
        lora_b=lora_b,
        head_weight=head_weight,
        head_bias=head_bias,
    )
    return RSFMFineTuneResult(bundle=bundle, loss_history=tuple(losses))


def load_rsfm_checkpoint(
    checkpoint_dir: PathLikeInput,
    *,
    backbone: RSFMBackbone,
) -> RSFMCheckpointBundle:
    """Load a saved :class:`RSFMCheckpointBundle` from ``checkpoint_dir``."""
    ckpt_path = Path(checkpoint_dir)
    manifest = json.loads((ckpt_path / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest["embed_dim"]) != backbone.embed_dim:
        raise ValueError(
            f"Checkpoint embed_dim ({manifest['embed_dim']}) does not match "
            f"backbone.embed_dim ({backbone.embed_dim})"
        )

    weights = np.load(ckpt_path / "weights.npz")
    return RSFMCheckpointBundle(
        backbone=backbone,
        class_values=tuple(int(c) for c in manifest["class_values"]),
        band_names=tuple(manifest["band_names"]),
        wavelengths_um=manifest["wavelengths_um"],
        mean=weights["mean"],
        std=weights["std"],
        lora_rank=int(manifest["lora_rank"]),
        lora_alpha=float(manifest["lora_alpha"]),
        lora_a=weights["lora_a"],
        lora_b=weights["lora_b"],
        head_weight=weights["head_weight"],
        head_bias=weights["head_bias"],
    )
