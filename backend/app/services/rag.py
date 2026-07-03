"""
RAG Pipeline
------------
Two retrieval modes, one generation step.

Retrieval:
  TF-IDF  -- cosine similarity on normalised term-frequency * inverse-document-frequency vectors.
              Deterministic, zero infrastructure, sub-millisecond. Good when query terms appear
              verbatim in the document (financial reports, technical docs).

  BM25    -- Okapi BM25. Improves on TF-IDF with non-linear term frequency saturation (repeated
              terms give diminishing returns) and document-length normalisation (longer chunks
              aren't penalised or rewarded purely for being long). Standard default in Elasticsearch,
              Lucene, and most production search systems. Generally outperforms TF-IDF on longer
              or variable-length chunks.

Both methods are lexical -- they match exact tokens, not semantic meaning. A query for "earnings"
won't match a chunk that only says "net income". For financial reports this is usually fine because
terminology is consistent. For noisy or multilingual documents, dense embedding retrieval
(OpenAI text-embedding-3-small, Cohere embed-v3) would improve recall at the cost of an API
dependency and a vector store.

Generation:
  Top-k retrieved passages are injected into a Claude API prompt. The model generates a grounded
  answer and is instructed to cite passage numbers explicitly.
"""

import asyncio
import os
import re
import math
import logging
from collections import defaultdict

import httpx

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4


# ── Retrieval: TF-IDF ─────────────────────────────────────────────────────────

class TFIDFIndex:
    """
    In-memory TF-IDF retriever with cosine similarity scoring.
    Score = dot(q_tfidf, chunk_tfidf) / (|q_tfidf| * |chunk_tfidf|) -- normalised [0, 1].
    Returned scores are multiplied by 100 for readability.
    """

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


# ── Retrieval: BM25 ──────────────────────────────────────────────────────────

class BM25Index:
    """
    Okapi BM25 retrieval.

    score(q, d) = sum_t [ IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl)) ]

    k1=1.5  term frequency saturation: extra occurrences of a term matter less as count grows
    b=0.75  length normalisation: longer chunks get penalised relative to average chunk length

    Returned scores are normalised to [0, 100] relative to the top-scoring chunk so they're
    comparable to the TF-IDF output in the API response.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[dict] = []
        self.doc_tf: list[dict[str, int]] = []
        self.doc_len: list[int] = []
        self.avg_dl: float = 1.0
        self.idf: dict[str, float] = {}
        self.N: int = 0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def build(self, chunks: list[str]) -> None:
        self.chunks = [{"text": c, "index": i} for i, c in enumerate(chunks)]
        self.N = len(chunks)
        token_lists = [self._tokenize(c) for c in chunks]

        self.doc_tf = []
        for tl in token_lists:
            freq: dict[str, int] = defaultdict(int)
            for t in tl:
                freq[t] += 1
            self.doc_tf.append(dict(freq))

        self.doc_len = [len(tl) for tl in token_lists]
        self.avg_dl = sum(self.doc_len) / self.N if self.N > 0 else 1.0

        doc_freq: dict[str, int] = defaultdict(int)
        for tl in token_lists:
            for t in set(tl):
                doc_freq[t] += 1

        # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1) -- avoids negative IDF for common terms
        self.idf = {
            t: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for t, df in doc_freq.items()
        }

    def query(self, q: str, k: int = TOP_K) -> list[dict]:
        q_tokens = self._tokenize(q)

        scores = []
        for i, tf in enumerate(self.doc_tf):
            dl = self.doc_len[i]
            score = 0.0
            for t in q_tokens:
                if t not in self.idf:
                    continue
                tf_val = tf.get(t, 0)
                numerator = tf_val * (self.k1 + 1)
                denominator = tf_val + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                score += self.idf[t] * (numerator / denominator)
            scores.append((score, i))

        scores.sort(reverse=True)
        max_score = scores[0][0] if scores and scores[0][0] > 0 else 1.0

        return [
            {
                "text": self.chunks[i]["text"],
                "score": round((s / max_score) * 100, 1),
                "index": i,
            }
            for s, i in scores[:k]
            if s > 0
        ]


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self):
        self._tfidf: dict[str, TFIDFIndex] = {}
        self._bm25: dict[str, BM25Index] = {}
        self._meta: dict[str, dict] = {}
        self._cache: dict[str, dict] = {}

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunks.append(" ".join(words[start:end]))
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def ingest(self, doc_id: str, text: str, title: str) -> int:
        chunks = self._chunk_text(text)

        tfidf = TFIDFIndex()
        tfidf.build(chunks)
        self._tfidf[doc_id] = tfidf

        bm25 = BM25Index()
        bm25.build(chunks)
        self._bm25[doc_id] = bm25

        self._meta[doc_id] = {"title": title, "chunks": len(chunks), "chars": len(text)}
        logger.info("Indexed '%s': %d chunks (TF-IDF + BM25)", title, len(chunks))
        return len(chunks)

    def delete(self, doc_id: str) -> None:
        self._tfidf.pop(doc_id, None)
        self._bm25.pop(doc_id, None)
        self._meta.pop(doc_id, None)

    def list_documents(self) -> list[dict]:
        return [{"id": k, **v} for k, v in self._meta.items()]

    def compare_retrieval(self, doc_id: str, question: str, k: int = TOP_K) -> dict:
        """
        Run both TF-IDF and BM25 on the same query and return a side-by-side comparison.
        Useful for understanding when the two methods disagree and which is more appropriate
        for a given question type.
        """
        if doc_id not in self._tfidf:
            return {"error": "Document not indexed"}

        tfidf_results = self._tfidf[doc_id].query(question, k=k)
        bm25_results = self._bm25[doc_id].query(question, k=k)

        tfidf_indices = {r["index"] for r in tfidf_results}
        bm25_indices = {r["index"] for r in bm25_results}
        overlap = tfidf_indices & bm25_indices

        return {
            "tfidf": tfidf_results,
            "bm25": bm25_results,
            "agreement": {
                "shared_chunks": len(overlap),
                "total_retrieved": k,
                "rate": round(len(overlap) / k, 2),
                "note": (
                    "High agreement means both methods rank the same passages. "
                    "Low agreement often indicates the query has synonyms or paraphrases "
                    "that one method handles better than the other."
                ),
            },
        }

    async def query(
        self,
        doc_id: str,
        question: str,
        retrieval_mode: str = "bm25",
    ) -> dict:
        if doc_id not in self._tfidf:
            return {"error": "Document not indexed", "answer": None, "retrieved": []}

        cache_key = f"{doc_id}:{retrieval_mode}:{question.lower().strip()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        index = self._bm25[doc_id] if retrieval_mode == "bm25" else self._tfidf[doc_id]
        retrieved = index.query(question, k=TOP_K)

        if not retrieved:
            return {"answer": "No relevant passages found.", "retrieved": [], "retrieval_mode": retrieval_mode}

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return {"error": "GROQ_API_KEY not set on server.", "answer": None, "retrieved": retrieved}

        context = "\n\n---\n\n".join(
            f"[Passage {i + 1}]\n{r['text']}" for i, r in enumerate(retrieved)
        )
        prompt = (
            "You are a financial analyst assistant. Answer the question using ONLY the "
            "passages provided. If the answer cannot be found in the passages, say so "
            "explicitly. Be specific and cite passage numbers where relevant.\n\n"
            f"PASSAGES:\n{context}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )

        resp = None
        for attempt in range(3):
            async with httpx.AsyncClient(timeout=40.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 1024,
                    },
                )
            if resp.status_code == 429 and attempt < 2:
                await asyncio.sleep(6)
                continue
            break

        if resp.status_code != 200:
            logger.warning("Groq API error %d: %s", resp.status_code, resp.text[:200])
            return {
                "error": f"Groq API returned {resp.status_code}",
                "answer": None,
                "retrieved": retrieved,
                "retrieval_mode": retrieval_mode,
            }

        data = resp.json()
        answer = data["choices"][0]["message"]["content"] if data.get("choices") else "No response"
        result = {
            "answer": answer,
            "retrieved": retrieved,
            "retrieval_mode": retrieval_mode,
            "model": "llama-3.1-8b-instant",
        }
        self._cache[cache_key] = result
        return result
