"""Stage-2: fine-grained multimodal re-ranking of the top-K Stage-1 candidates."""

from multiscore.stage2.cot_score import BidirectionalCoTScorer, cot_score
from multiscore.stage2.qa_score import QAPair, QARelevanceScorer, parse_qa_pairs, qa_score
from multiscore.stage2.rerank import Stage2Output, combine_scores, rerank

__all__ = [
    "BidirectionalCoTScorer",
    "QAPair",
    "QARelevanceScorer",
    "Stage2Output",
    "combine_scores",
    "cot_score",
    "parse_qa_pairs",
    "qa_score",
    "rerank",
]
