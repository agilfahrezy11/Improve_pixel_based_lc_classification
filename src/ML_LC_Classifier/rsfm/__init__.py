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
from .geobia import (
    assign_segment_labels,
    pool_rsfm_by_segments,
    project_segment_probabilities,
)
from .inference import infer_rsfm_probabilities

__all__ = [
    "OpenGeoAIBackboneAdapter",
    "RSFMBackbone",
    "RSFMCheckpointBundle",
    "RSFMChipDataset",
    "RSFMFineTuneResult",
    "SimpleCPUBackbone",
    "assign_segment_labels",
    "extract_rsfm_embeddings",
    "finetune_rsfm",
    "infer_rsfm_probabilities",
    "list_opengeoai_models",
    "load_opengeoai_backbone",
    "load_rsfm_checkpoint",
    "pool_rsfm_by_segments",
    "prepare_rsfm_chips",
    "project_segment_probabilities",
]
