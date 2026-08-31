"""Every prompt used by MULTI-SCORE, in one place.

The captioning prompts and the QA-generation prompt are reproduced verbatim
from the appendix of the paper.  The Bidirectional-CoT templates follow the
description in Section 3.2 and Figure 3: two in-context demonstrations (the
ablation's optimum), an explicit alignment rationale, and a terminal ``<emb>``
token whose preceding hidden state is used as the embedding.
"""

from __future__ import annotations

from typing import List

EMB_TOKEN = "<emb>"

# --------------------------------------------------------------------- #
# Offline pre-processing: modality -> text (Appendix A.4)
# --------------------------------------------------------------------- #
CAPTION_IMAGE = """You are given an image: <i>
Generate a detailed caption describing the key objects, their attributes, spatial layout, and interactions.
Include information about the scene type, context, and any salient visual cues that convey intent or activity.
Output a detailed caption that captures the essential meaning of the image."""

CAPTION_IMAGE_TEXT = """You are given an image and its accompanying text: (<i>, <t>)
Generate a unified caption that integrates both visual and textual information.
Describe how the image and text complement each other.
Mention entities, actions, and context shared across the image and text.
Output a detailed and coherent caption that captures the essential, combined meaning of the image and text."""

CAPTION_VIDEO = """You are given a short video clip: <v>
Generate a temporally aware caption describing the sequence of actions and events.
Mention key objects, subjects, and transitions over time, emphasizing movement and interactions.
Output a detailed caption that captures the essential meaning of the video."""

CAPTION_AUDIO = """You are given an audio clip: <a>
Generate a descriptive caption summarizing the content of the audio.
Include information about sound types, speakers, emotions, acoustic events, and temporal changes.
Output a detailed caption that captures the essential meaning of the audio."""

CAPTION_AUDIO_VIDEO = """You are given synchronized audio and video data: (<a>, <v>)
Generate a unified caption that combines both auditory and visual cues.
Describe the main event or scene, integrating spoken words, sounds, and visible actions over time.
Output a detailed and coherent caption that captures the essential, combined meaning of the audio and video."""

# Modality key -> (prompt, captioning model used in the paper)
CAPTION_PROMPTS = {
    "i": (CAPTION_IMAGE, "qwen3-vl-8b"),
    "it": (CAPTION_IMAGE_TEXT, "qwen3-vl-8b"),
    "v": (CAPTION_VIDEO, "qwen3-vl-8b"),
    "a": (CAPTION_AUDIO, "qwen2-audio-7b"),
    "av": (CAPTION_AUDIO_VIDEO, "qwen2.5-omni-7b"),
}


def caption_prompt(modality: str) -> str:
    """Look up the captioning prompt for a modality key (``i``, ``it``, ``v``, ``a``, ``av``)."""

    key = modality.replace(",", "").replace("(", "").replace(")", "").replace("+", "").strip().lower()
    if key not in CAPTION_PROMPTS:
        raise KeyError(f"no caption prompt for modality '{modality}'; have {sorted(CAPTION_PROMPTS)}")
    return CAPTION_PROMPTS[key][0]


# --------------------------------------------------------------------- #
# Stage-2a: Question Answering Relevance Score (Eq. 8)
# --------------------------------------------------------------------- #
QA_GENERATION_PROMPT = """You are given a multimodal query: {query}
Generate {num_questions} Yes/No questions that capture essential semantics from the above query.
For each question, also provide its correct Yes/No answer. Format:
Q1:..? A: Yes/No; Q2:..? A: Yes/No; .."""

QA_ANSWER_PROMPT = """You are given a database candidate as context.

Candidate: {candidate}

Using only the information in the candidate, answer the following question.
Answer with only Yes or No.
{question}"""


def qa_generation_prompt(query: str, num_questions: int = 7) -> str:
    return QA_GENERATION_PROMPT.format(query=query, num_questions=num_questions)


def qa_answer_prompt(candidate: str, question: str) -> str:
    return QA_ANSWER_PROMPT.format(candidate=candidate, question=question)


# --------------------------------------------------------------------- #
# Stage-2b: Bidirectional-CoT Embedding Score (Eq. 7)
# --------------------------------------------------------------------- #
COT_INSTRUCTION = (
    "You judge how well a retrieval query and a database candidate align.\n"
    "Reason step by step: list the concepts in the source, check whether each "
    "one is supported by the target, and conclude with the alignment.\n"
    "If every concept in the source is found in the target, the alignment is "
    "high; if concepts are missing or contradicted, the alignment is low.\n"
    "Summarise the alignment in one word."
)

#: Two in-context demonstrations -- the paper's ablation finds 2 optimal.
#: Each carries a rationale for both directions, since the q2c and c2q prompts
#: reason about different source sets.
COT_EXAMPLES: List[dict] = [
    {
        "query": "a person is swimming in some white water rapids",
        "candidate": (
            "A kayaker paddles hard through churning white water rapids while a "
            "swimmer in a red life vest is carried downstream beside the boat."
        ),
        "q2c_reasoning": (
            "Concepts in the query: (1) a person, (2) swimming, (3) white water "
            "rapids. The candidate shows a swimmer, so (1) holds; the swimmer is "
            "in the water being carried downstream, so (2) holds; the water is "
            "explicitly white water rapids, so (3) holds. Every query concept is "
            "supported by the candidate."
        ),
        "c2q_reasoning": (
            "Concepts in the candidate: (1) a kayaker, (2) white water rapids, "
            "(3) a swimmer in a life vest. The query asks for a person swimming "
            "in white water rapids, which covers (2) and (3); the extra kayak "
            "detail refines the scene rather than contradicting it. The candidate "
            "is about what the query asks for."
        ),
        "verdict": "high",
    },
    {
        "query": "a person is swimming in some white water rapids",
        "candidate": (
            "A calm lake at sunset with two anglers seated in a moored rowing "
            "boat; no one is in the water."
        ),
        "q2c_reasoning": (
            "Concepts in the query: (1) a person, (2) swimming, (3) white water "
            "rapids. People are present, so (1) holds; nobody is in the water, so "
            "(2) fails; the water is a calm lake rather than rapids, so (3) fails. "
            "Most query concepts are unsupported."
        ),
        "c2q_reasoning": (
            "Concepts in the candidate: (1) a calm lake at sunset, (2) two "
            "anglers, (3) a moored boat, (4) nobody in the water. The query asks "
            "for swimming in white water rapids, which none of these support and "
            "(4) directly contradicts."
        ),
        "verdict": "low",
    },
]


def _render_examples(num_examples: int, reversed_direction: bool) -> str:
    """Render the in-context demonstrations for one direction.

    In the c2q direction the candidate is presented first and the rationale
    reasons from candidate concepts to the query -- not merely a reordering of
    the q2c block, because alignment is asymmetric.
    """

    source_role, target_role = ("Candidate", "Query") if reversed_direction else ("Query", "Candidate")
    source_key, target_key = ("candidate", "query") if reversed_direction else ("query", "candidate")
    reasoning_key = "c2q_reasoning" if reversed_direction else "q2c_reasoning"

    blocks = []
    for example in COT_EXAMPLES[: max(0, num_examples)]:
        blocks.append(
            "{sr}: {source}\n{tr}: {target}\nReasoning: {reasoning}\nAlignment: {verdict}".format(
                sr=source_role,
                tr=target_role,
                source=example[source_key],
                target=example[target_key],
                reasoning=example[reasoning_key],
                verdict=example["verdict"],
            )
        )
    return "\n\n".join(blocks)


def _cot_prompt(
    source: str,
    target: str,
    num_examples: int,
    emb_token: str,
    reversed_direction: bool,
) -> str:
    source_role, target_role = ("Candidate", "Query") if reversed_direction else ("Query", "Candidate")
    parts = [COT_INSTRUCTION]
    examples = _render_examples(num_examples, reversed_direction)
    if examples:
        parts.append(examples)
    parts.append(
        "{sr}: {source}\n{tr}: {target}\nReasoning:".format(
            sr=source_role, tr=target_role, source=source, target=target
        )
    )
    # The marker terminates the prompt; the hidden state *preceding* it is the
    # embedding (one-word limitation, cf. Jiang et al., 2024).
    return "\n\n".join(parts) + f"\nAlignment in one word: {emb_token}"


def query_to_candidate_prompt(
    query: str, candidate: str, num_examples: int = 2, emb_token: str = EMB_TOKEN
) -> str:
    """q2c prompt -> hidden state ``z_{q2c}``."""

    return _cot_prompt(query, candidate, num_examples, emb_token, reversed_direction=False)


def candidate_to_query_prompt(
    candidate: str, query: str, num_examples: int = 2, emb_token: str = EMB_TOKEN
) -> str:
    """c2q prompt -> hidden state ``z_{c2q}`` (roles swapped, not just reordered)."""

    return _cot_prompt(candidate, query, num_examples, emb_token, reversed_direction=True)


__all__ = [
    "CAPTION_AUDIO",
    "CAPTION_AUDIO_VIDEO",
    "CAPTION_IMAGE",
    "CAPTION_IMAGE_TEXT",
    "CAPTION_PROMPTS",
    "CAPTION_VIDEO",
    "COT_EXAMPLES",
    "COT_INSTRUCTION",
    "EMB_TOKEN",
    "QA_ANSWER_PROMPT",
    "QA_GENERATION_PROMPT",
    "candidate_to_query_prompt",
    "caption_prompt",
    "qa_answer_prompt",
    "qa_generation_prompt",
    "query_to_candidate_prompt",
]
