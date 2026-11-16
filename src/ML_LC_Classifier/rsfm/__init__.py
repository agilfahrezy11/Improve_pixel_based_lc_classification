"""Remote Sensing Foundation Model (RSFM) subpackage."""

from .backbone import (
    OpenGeoAIBackboneAdapter,
    RSFMBackbone,
    SimpleCPUBackbone,
    list_opengeoai_models,
    load_opengeoai_backbone,
)
from .embedding_stack import extract_rsfm_embeddings

__all__ = [
    "OpenGeoAIBackboneAdapter",
    "RSFMBackbone",
    "SimpleCPUBackbone",
    "extract_rsfm_embeddings",
    "list_opengeoai_models",
    "load_opengeoai_backbone",
]
