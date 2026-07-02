import pytest
from app.services.rag import TFIDFIndex, BM25Index, RAGPipeline

# Five chunks that cover distinct financial topics
CHUNKS = [
    "Revenue grew 15 percent in fiscal year 2023 driven by digital services expansion.",
    "Net profit after tax increased to SGD 2.1 billion, up 8 percent year-on-year.",
    "The board declared a final dividend of 36 cents per share for shareholders.",
    "Operating expenses rose due to higher staff costs and technology investment.",
    "Customer deposits reached a record SGD 450 billion at end of December 2023.",
]
FULL_TEXT = " ".join(CHUNKS)


# ── TF-IDF ────────────────────────────────────────────────────────────────────

class TestTFIDFIndex:
    def setup_method(self):
        self.idx = TFIDFIndex()
        self.idx.build(CHUNKS)

    def test_known_term_returns_results(self):
        results = self.idx.query("revenue")
        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_relevant_chunk_scores_highest(self):
        results = self.idx.query("dividend shareholders")
        assert "dividend" in results[0]["text"].lower()

    def test_unknown_term_returns_empty(self):
        results = self.idx.query("xyzqwerty")
        assert results == []

    def test_scores_bounded_0_to_100(self):
        results = self.idx.query("profit net income deposits")
        for r in results:
            assert 0 <= r["score"] <= 100

    def test_result_has_required_keys(self):
        results = self.idx.query("revenue")
        assert {"text", "score", "index"} <= set(results[0].keys())

    def test_top_k_respected(self):
        results = self.idx.query("the", k=2)
        assert len(results) <= 2

    def test_scores_descending(self):
        results = self.idx.query("profit expenses revenue")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ── BM25 ──────────────────────────────────────────────────────────────────────

class TestBM25Index:
    def setup_method(self):
        self.idx = BM25Index()
        self.idx.build(CHUNKS)

    def test_known_term_returns_results(self):
        results = self.idx.query("dividend")
        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_relevant_chunk_scores_highest(self):
        results = self.idx.query("dividend shareholders")
        assert "dividend" in results[0]["text"].lower()

    def test_unknown_term_returns_empty(self):
        results = self.idx.query("xyzqwerty")
        assert results == []

    def test_scores_bounded_0_to_100(self):
        results = self.idx.query("profit year deposits")
        for r in results:
            assert 0 <= r["score"] <= 100

    def test_top_chunk_scores_100(self):
        # The highest-scoring chunk is always normalised to 100
        results = self.idx.query("revenue digital services")
        assert results[0]["score"] == 100.0

    def test_top_k_respected(self):
        results = self.idx.query("the", k=2)
        assert len(results) <= 2

    def test_scores_descending(self):
        results = self.idx.query("profit expenses revenue")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_length_normalisation_applied(self):
        # BM25 uses avg_dl; verify the index stores non-zero avg_dl after build
        assert self.idx.avg_dl > 0

    def test_k1_and_b_defaults(self):
        assert self.idx.k1 == 1.5
        assert self.idx.b == 0.75


# ── RAGPipeline ───────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def setup_method(self):
        self.pipeline = RAGPipeline()
        self.pipeline.ingest("doc1", FULL_TEXT, "Test Document")

    def test_list_documents_after_ingest(self):
        docs = self.pipeline.list_documents()
        assert any(d["id"] == "doc1" for d in docs)

    def test_ingest_returns_chunk_count(self):
        count = self.pipeline.ingest("doc2", "short text for testing", "Short Doc")
        assert count >= 1

    def test_both_indexes_built_after_ingest(self):
        assert "doc1" in self.pipeline._tfidf
        assert "doc1" in self.pipeline._bm25

    def test_compare_returns_agreement_rate(self):
        result = self.pipeline.compare_retrieval("doc1", "revenue profit")
        assert "tfidf" in result
        assert "bm25" in result
        assert "agreement" in result
        assert 0 <= result["agreement"]["rate"] <= 1

    def test_compare_missing_doc_returns_error(self):
        result = self.pipeline.compare_retrieval("nonexistent", "query")
        assert "error" in result

    def test_delete_removes_document(self):
        self.pipeline.ingest("temp", "temporary document for deletion testing", "Temp")
        self.pipeline.delete("temp")
        ids = [d["id"] for d in self.pipeline.list_documents()]
        assert "temp" not in ids
        assert "doc1" in ids
