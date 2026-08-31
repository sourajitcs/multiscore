"""Stage-2: prompt construction, QA parsing, the two scores and their mixture."""

from __future__ import annotations

import numpy as np
import pytest

from multiscore.config import Stage2Config
from multiscore.models.stub import LexicalMLLM
from multiscore.stage2.cot_score import BidirectionalCoTScorer, cosine
from multiscore.stage2.prompts import (
    EMB_TOKEN,
    candidate_to_query_prompt,
    caption_prompt,
    qa_generation_prompt,
    query_to_candidate_prompt,
)
from multiscore.stage2.qa_score import QAPair, QARelevanceScorer, normalize_yes_no, parse_qa_pairs
from multiscore.stage2.rerank import combine_scores, rerank


# ---------------------------------------------------------------- prompts --
def test_cot_prompts_end_with_the_marker():
    for prompt in (
        query_to_candidate_prompt("a red bus", "a photo of a red bus"),
        candidate_to_query_prompt("a photo of a red bus", "a red bus"),
    ):
        assert prompt.rstrip().endswith(EMB_TOKEN)


def test_cot_prompts_are_direction_specific():
    q2c = query_to_candidate_prompt("A", "B")
    c2q = candidate_to_query_prompt("B", "A")
    assert q2c != c2q
    assert q2c.rindex("Query: A") < q2c.rindex("Candidate: B")
    assert c2q.rindex("Candidate: B") < c2q.rindex("Query: A")


def test_number_of_cot_examples_is_configurable():
    zero = query_to_candidate_prompt("A", "B", num_examples=0)
    two = query_to_candidate_prompt("A", "B", num_examples=2)
    assert len(two) > len(zero)
    assert two.count("Alignment:") == 2


def test_cot_examples_reason_in_both_directions():
    q2c = query_to_candidate_prompt("A", "B")
    c2q = candidate_to_query_prompt("B", "A")
    assert "Concepts in the query" in q2c
    assert "Concepts in the candidate" in c2q


def test_caption_prompts_cover_every_modality():
    for modality in ("i", "it", "v", "a", "av"):
        assert caption_prompt(modality)
    with pytest.raises(KeyError):
        caption_prompt("nope")


def test_qa_generation_prompt_carries_the_budget():
    prompt = qa_generation_prompt("a dog on a skateboard", num_questions=5)
    assert "5 Yes/No questions" in prompt
    assert "a dog on a skateboard" in prompt


# ------------------------------------------------------------------- QA ----
def test_parse_qa_pairs_handles_the_documented_format():
    raw = "Q1: Is there a dog? A: Yes; Q2: Is it raining? A: No;"
    pairs = parse_qa_pairs(raw)
    assert [p.answer for p in pairs] == ["Yes", "No"]
    assert pairs[0].question == "Is there a dog?"


def test_parse_qa_pairs_handles_newlines_and_caps():
    raw = "Q1. Is the sky blue?\nA: yes\nQ2. Is it night?\nA: NO"
    pairs = parse_qa_pairs(raw)
    assert len(pairs) == 2
    assert pairs[1].answer == "No"


def test_parse_qa_pairs_respects_max_pairs():
    raw = " ".join(f"Q{i}: q{i}? A: Yes;" for i in range(10))
    assert len(parse_qa_pairs(raw, max_pairs=3)) == 3


def test_normalize_yes_no_is_defensive():
    assert normalize_yes_no("Yes, definitely") == "Yes"
    assert normalize_yes_no(" no.") == "No"
    assert normalize_yes_no("¯\\_(ツ)_/¯") == "No"


def test_qa_score_is_accuracy_over_questions():
    backend = LexicalMLLM()
    scorer = QARelevanceScorer(backend, Stage2Config(num_questions=2))
    pairs = [QAPair("Is there a dog?", "Yes"), QAPair("Is there a piano?", "Yes")]
    scores = scorer.score_candidates(
        "a dog", ["a dog running in a park", "a piano in a concert hall"], qa_pairs=pairs
    )
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_qa_logs_explain_the_decision():
    backend = LexicalMLLM()
    scorer = QARelevanceScorer(backend, Stage2Config(num_questions=1))
    scorer.score_candidates(
        "a dog", ["a dog in a park"], qa_pairs=[QAPair("Is there a dog?", "Yes")]
    )
    log = scorer.last_logs[0]
    assert log.questions and log.gold and log.predicted
    assert "S_QA" in log.explain()


def test_no_questions_yields_uninformative_scores():
    backend = LexicalMLLM()
    scorer = QARelevanceScorer(backend, Stage2Config())
    scores = scorer.score_candidates("q", ["a", "b"], qa_pairs=[])
    assert np.allclose(scores, 0.0)


# ------------------------------------------------------------------ CoT ----
def test_cot_score_is_a_cosine_in_range():
    scorer = BidirectionalCoTScorer(LexicalMLLM(), Stage2Config())
    scores = scorer.score_candidates("a red bus", ["a red bus at a stop", "a green field"])
    assert scores.shape == (2,)
    assert np.all(scores >= -1.0 - 1e-6) and np.all(scores <= 1.0 + 1e-6)


def test_cosine_handles_degenerate_vectors():
    assert cosine(np.zeros(4), np.ones(4)) == 0.0
    assert pytest.approx(cosine(np.ones(4), np.ones(4)), abs=1e-6) == 1.0


# --------------------------------------------------------------- combine ---
def test_combine_scores_is_convex():
    cot = np.array([1.0, 0.0], dtype=np.float32)
    qa = np.array([0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(combine_scores(cot, qa, 0.6), [0.6, 0.4], atol=1e-6)
    np.testing.assert_allclose(combine_scores(cot, qa, 1.0), cot)
    np.testing.assert_allclose(combine_scores(cot, qa, 0.0), qa)


def test_combine_scores_validates_alpha_and_shapes():
    with pytest.raises(ValueError):
        combine_scores(np.zeros(2), np.zeros(2), 1.5)
    with pytest.raises(ValueError):
        combine_scores(np.zeros(2), np.zeros(3), 0.5)


def test_rerank_orders_by_the_combined_score():
    output = rerank(
        "a dog running in a park",
        ["a cat on a sofa", "a dog running in a park at sunrise", "a plate of pasta"],
        backend=LexicalMLLM(),
        config=Stage2Config(num_questions=4),
    )
    assert len(output.indices) == 3
    assert np.all(np.diff(output.scores) <= 1e-6)
    assert set(output.ids) == {"0", "1", "2"}


def test_disabling_a_component_moves_alpha_to_the_other():
    candidates = ["a dog", "a cat"]
    cot_only = rerank("a dog", candidates, LexicalMLLM(), Stage2Config(enable_qa=False))
    qa_only = rerank("a dog", candidates, LexicalMLLM(), Stage2Config(enable_cot=False))
    np.testing.assert_allclose(cot_only.scores, cot_only.cot_scores, atol=1e-6)
    np.testing.assert_allclose(qa_only.scores, qa_only.qa_scores, atol=1e-6)
