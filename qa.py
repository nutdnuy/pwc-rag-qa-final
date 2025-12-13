#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _word_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    start_char: int
    end_char: int
    text: str


def chunk_document(text: str, chunk_words: int = 250, overlap_words: int = 60) -> List[Chunk]:
    spans = _word_spans(text)
    chunks: List[Chunk] = []
    step = max(1, chunk_words - overlap_words)

    for i in range(0, len(spans), step):
        j = min(i + chunk_words, len(spans))
        start_char = spans[i][0]
        end_char = spans[j - 1][1]
        chunks.append(
            Chunk(
                chunk_id=len(chunks),
                start_char=start_char,
                end_char=end_char,
                text=text[start_char:end_char].strip(),
            )
        )
        if j == len(spans):
            break
    return chunks


@dataclass
class Index:
    vectorizer: TfidfVectorizer
    matrix: np.ndarray
    chunks: List[Chunk]


def build_index(chunks: List[Chunk]) -> Index:
    max_df = 1.0 if len(chunks) <= 1 else 0.98
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=max_df,
        stop_words="english",
    )
    X = vec.fit_transform([c.text for c in chunks])
    return Index(vec, X, chunks)


def retrieve(index: Index, question: str, top_k: int = 3):
    qv = index.vectorizer.transform([question])
    sims = cosine_similarity(qv, index.matrix)[0]
    idx = np.argsort(-sims)[:top_k]
    return [(index.chunks[i], float(sims[i])) for i in idx]


def extractive_answer(index: Index, question: str, chunk: Chunk):
    sentences = []
    for s in _SENT_SPLIT.split(chunk.text):
        s = s.strip()
        if not s:
            continue
        rel = chunk.text.find(s)
        if rel != -1:
            sentences.append((s, chunk.start_char + rel, chunk.start_char + rel + len(s)))

    if not sentences:
        return "", chunk.start_char, chunk.end_char, 0.0

    qv = index.vectorizer.transform([question])
    sv = index.vectorizer.transform([s[0] for s in sentences])
    sims = cosine_similarity(qv, sv)[0]
    i = int(np.argmax(sims))
    return (*sentences[i], float(sims[i]))


def answer_question(
    policy_path: Path,
    question: str,
    top_k: int = 3,
    refuse_threshold: float = 0.12,
):
    text = policy_path.read_text(encoding="utf-8")
    chunks = chunk_document(text)
    index = build_index(chunks)
    retrieved = retrieve(index, question, top_k)

    best_chunk, best_score = retrieved[0]
    ans, a, b, sent_score = extractive_answer(index, question, best_chunk)

    refused = best_score < refuse_threshold or not ans.strip()

    return {
        "question": question,
        "refused": refused,
        "answer": None if refused else ans,
        "confidence": {
            "best_chunk_similarity": best_score,
            "best_sentence_similarity": sent_score,
        },
        "citations": {
            "best_chunk": {
                "chunk_id": best_chunk.chunk_id,
                "start_char": best_chunk.start_char,
                "end_char": best_chunk.end_char,
            },
            "best_sentence": {
                "start_char": a,
                "end_char": b,
            },
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Lightweight RAG Q&A (no LLM)")
    ap.add_argument("--question", required=True, help='e.g. "What are the core working hours?"')
    ap.add_argument("--policy", default="policy.txt", help="Path to policy.txt")
    ap.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    ap.add_argument("--refuse_threshold", type=float, default=0.12, help="Refuse if best chunk similarity < threshold")
    ap.add_argument("--json", action="store_true", help="Print full JSON output")
    ap.add_argument("--show_confidence", action="store_true", help="Show confidence scores in human output")
    ap.add_argument("--show_citations", action="store_true", help="Show chunk id and character offsets in human output")
    args = ap.parse_args()

    res = answer_question(
        Path(args.policy),
        args.question,
        top_k=args.top_k,
        refuse_threshold=args.refuse_threshold,
    )

    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    print(f"Q: {res['question']}")
    if res["refused"]:
        print("A: [REFUSED] Low relevance to the provided document.")
    else:
        print(f"A: {res['answer']}")

    if args.show_confidence:
        conf = res["confidence"]
        print(f"Confidence: best_chunk_similarity={conf['best_chunk_similarity']:.3f}, best_sentence_similarity={conf['best_sentence_similarity']:.3f}")

    if args.show_citations:
        bc = res["citations"]["best_chunk"]
        bs = res["citations"]["best_sentence"]
        print(f"Citations: best_chunk_id={bc['chunk_id']} chunk_chars=[{bc['start_char']},{bc['end_char']}) answer_chars=[{bs['start_char']},{bs['end_char']})")


if __name__ == "__main__":
    main()
