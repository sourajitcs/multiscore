"""MULTI-SCORE: Zero-Shot Multimodal Retrieval with Multi-Scale Contextual Representations.

The package is organised around the two stages of the paper:

* ``multiscore.stage1`` -- *Pyramid Rank*: admissible, epsilon-bounded candidate
  filtering over Matryoshka (MRL) embedding levels.
* ``multiscore.stage2`` -- *Bidirectional-CoT Embedding Score* and *Question
  Answering Relevance Score*, combined into a single re-ranking score.

``multiscore.pipeline.MultiScore`` glues both stages together.
"""

from multiscore.config import (
    MRLConfig,
    MultiScoreConfig,
    Stage1Config,
    Stage2Config,
    load_config,
)
from multiscore.pipeline import MultiScore, RetrievalResult
from multiscore.stage1.pyramid_rank import Stage1Output, pyramid_rank
from multiscore.stage1.upper_bound import similarity_upper_bound
from multiscore.stage2.rerank import combine_scores

__version__ = "0.1.0"

__all__ = [
    "MRLConfig",
    "MultiScore",
    "MultiScoreConfig",
    "RetrievalResult",
    "Stage1Config",
    "Stage1Output",
    "Stage2Config",
    "combine_scores",
    "load_config",
    "pyramid_rank",
    "similarity_upper_bound",
    "__version__",
]
