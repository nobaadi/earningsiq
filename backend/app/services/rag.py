"""
RAG Pipeline
------------
1. Ingest:    Split document into overlapping chunks
2. Index:     Build TF-IDF term matrix for each document in memory
3. Retrieve:  Score chunks by cosine similarity against the query vector
4. Generate:  Send top-k chunks as context to Claude API, return grounded answer

Why TF-IDF instead of a vector database?
  - Zero infrastructure: no Pinecone, Weaviate, or Postgres pgvector needed
  - Fully explainable: every retrieval score is a deterministic dot product
  - Fast enough: sub-millisecond retrieval for documents up to ~500k words
  - Interview-friendly: the math is on one page -- TF * IDF, cosine similarity

The tradeoff is semantic recall. TF-IDF misses synonyms and paraphrases that
dense vector embeddings would catch. For financial report Q&A where terminology
is consistent (revenue, margin, EPS appear verbatim), lexical matching is
sufficient and the simpler system is easier to reason about and debug.
"""

import re
import math
import logging
from collections import defaultdict

import httpx

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4


class TFIDFIndex:
    """In-memory TF-IDF retriever. No external dependencies."""

    def __init__(self):
        self.chunks: list[dict] = []
        self.tf_matrix: list[dict] = []
        self.idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def _tf(self, tokens: list[str]) -> dict[str, float]:
        freq: dict[str, int] = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        total = len(tokens) or 1
        return {t: c / total for t, c in freq.items()}

    def build(self, chunks: list[str]) -> None:
        self.chunks = [{"text": c, "index": i} for i, c in enumerate(chunks)]
        token_lists = [self._tokenize(c) for c in chunks]
        self.tf_matrix = [self._tf(tl) for tl in token_lists]
        N = len(chunks)
        doc_freq: dict[str, int] = defaultdict(int)
        for tl in token_lists:
            for t in set(tl):
                doc_freq[t] += 1
        # Add-1 smoothing so unseen terms don't cause division by zero
        self.idf = {t: math.log((N + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}

    def query(self, q: str, k: int = TOP_K) -> list[dict]:
        q_tokens = self._tokenize(q)
        q_tf = self._tf(q_tokens)
        q_vec = {t: q_tf[t] * self.idf.get(t, 0.0) for t in q_tokens}
        q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values())) or 1.0

        scores = []
        for i, tf in enumerate(self.tf_matrix):
            chunk_tfidf = {t: tf.get(t, 0.0) * self.idf.get(t, 0.0) for t in q_vec}
            dot = sum(q_vec[t] * chunk_tfidf[t] for t in q_vec)
            c_norm = math.sqrt(sum(v ** 2 for v in chunk_tfidf.values())) or 1.0
            scores.append((dot / (q_norm * c_norm), i))

        scores.sort(reverse=True)
        return [
            {"text": self.chunks[i]["text"], "score": round(s * 100, 1), "index": i}
            for s, i in scores[:k]
            if s > 0
        ]


class RAGPipeline:
    def __init__(self):
        self._indexes: dict[str, TFIDFIndex] = {}
        self._meta: dict[str, dict] = {}

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunks.append(" ".join(words[start:end]))
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def ingest(self, doc_id: str, text: str, title: str) -> int:
        chunks = self._chunk_text(text)
        idx = TFIDFIndex()
        idx.build(chunks)
        self._indexes[doc_id] = idx
        self._meta[doc_id] = {"title": title, "chunks": len(chunks), "chars": len(text)}
        logger.info("Indexed '%s': %d chunks", title, len(chunks))
        return len(chunks)

    def delete(self, doc_id: str) -> None:
        self._indexes.pop(doc_id, None)
        self._meta.pop(doc_id, None)

    def list_documents(self) -> list[dict]:
        return [{"id": k, **v} for k, v in self._meta.items()]

    async def query(self, doc_id: str, question: str, api_key: str) -> dict:
        if doc_id not in self._indexes:
            return {"error": "Document not indexed", "answer": None, "retrieved": []}

        retrieved = self._indexes[doc_id].query(question, k=TOP_K)
        if not retrieved:
            return {
                "answer": "No relevant passages found for this question.",
                "retrieved": [],
            }

        context = "\n\n---\n\n".join(
            f"[Passage {i + 1}]\n{r['text']}" for i, r in enumerate(retrieved)
        )

        prompt = (
            "You are a financial analyst assistant. Answer the question using ONLY the "
            "passages provided. If the answer cannot be found in the passages, say so "
            "explicitly. Be specific and cite passage numbers where relevant.\n\n"
            f"PASSAGES:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 800,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

        if resp.status_code != 200:
            error_body = resp.text[:200]
            logger.warning("Claude API error %d: %s", resp.status_code, error_body)
            return {
                "error": f"Claude API returned {resp.status_code}",
                "answer": None,
                "retrieved": retrieved,
            }

        data = resp.json()
        answer = data["content"][0]["text"] if data.get("content") else "No response"
        return {
            "answer": answer,
            "retrieved": retrieved,
            "model": "claude-haiku-4-5-20251001",
        }
