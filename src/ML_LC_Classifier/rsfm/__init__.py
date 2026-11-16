"""Remote Sensing Foundation Model (RSFM) subpackage."""

from .backbone import (
    OpenGeoAIBackboneAdapter,
    RSFMBackbone,
    SimpleCPUBackbone,
    list_opengeoai_models,
    load_opengeoai_backbone,
)
from .chips import RSFMChipDataset, prepare_rsfm_chips
from .embedding_stack import extract_rsfm_embeddings
from .finetune import (
    RSFMCheckpointBundle,
    RSFMFineTuneResult,
    finetune_rsfm,
    load_rsfm_checkpoint,
)

__all__ = [
    "OpenGeoAIBackboneAdapter",
    "RSFMBackbone",
    "RSFMCheckpointBundle",
    "RSFMChipDataset",
    "RSFMFineTuneResult",
    "SimpleCPUBackbone",
    "extract_rsfm_embeddings",
    "finetune_rsfm",
    "list_opengeoai_models",
    "load_opengeoai_backbone",
    "load_rsfm_checkpoint",
    "prepare_rsfm_chips",
]
