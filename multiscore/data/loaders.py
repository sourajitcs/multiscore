"""JSONL loaders for queries, candidates and qrels."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Set

from multiscore.data.schema import Candidate, Query
from multiscore.utils.io import read_jsonl


def load_queries(path: str, limit: Optional[int] = None) -> List[Query]:
    """Load ``queries.jsonl``; each row must carry ``id`` and (usually) ``positives``."""

    queries: List[Query] = []
    for row in read_jsonl(path):
        queries.append(Query.from_dict(row))
        if limit is not None and len(queries) >= limit:
            break
    return queries


def load_candidates(path: str, limit: Optional[int] = None) -> List[Candidate]:
    """Load ``candidates.jsonl`` -- the retrieval database for one benchmark."""

    candidates: List[Candidate] = []
    for row in read_jsonl(path):
        candidates.append(Candidate.from_dict(row))
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def load_qrels(path: str) -> Dict[str, Set[str]]:
    """Load relevance judgements.

    Accepts either JSONL (``{"query_id": ..., "candidate_id": ...}``) or the
    TREC-style whitespace format ``qid _ did rel``.
    """

    qrels: Dict[str, Set[str]] = {}
    if path.endswith((".jsonl", ".jsonl.gz")):
        for row in read_jsonl(path):
            qid = str(row["query_id"])
            did = str(row.get("candidate_id", row.get("doc_id")))
            if int(row.get("relevance", 1)) > 0:
                qrels.setdefault(qid, set()).add(did)
        return qrels

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            qid, did = fields[0], fields[2]
            relevance = int(fields[3]) if len(fields) > 3 else 1
            if relevance > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def attach_qrels(queries: Sequence[Query], qrels: Dict[str, Set[str]]) -> None:
    """Fill in ``Query.positives`` from a separate qrels file, in place."""

    for query in queries:
        if query.id in qrels:
            query.positives = sorted(qrels[query.id])


def resolve_split(root: str, dataset: str, task: str, split: str = "test") -> Dict[str, str]:
    """Conventional on-disk layout: ``<root>/<dataset>/<task>/<split>.*.jsonl``.

    ``docs/DATA.md`` describes how to produce it; no path is created or
    downloaded here.
    """

    base = os.path.join(root, dataset, task)
    return {
        "queries": os.path.join(base, f"{split}.queries.jsonl"),
        "candidates": os.path.join(base, "candidates.jsonl"),
        "qrels": os.path.join(base, f"{split}.qrels.jsonl"),
    }
