# EarningsIQ: Financial Report Q&A with RAG

A retrieval-augmented generation system for querying annual reports. Select a document, ask a question in plain English, and get a grounded answer citing the specific passages it was drawn from.

Two retrieval algorithms are implemented and can be compared side-by-side: TF-IDF cosine similarity and Okapi BM25. A dedicated `/api/compare` endpoint runs both on the same query and returns an agreement rate, making the retrieval behaviour directly observable.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Claude](https://img.shields.io/badge/Claude-Haiku_4.5-orange)](https://anthropic.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)

---

## What It Does

The backend implements a four-step RAG pipeline:

1. **Ingest** -- document text is split into 500-word overlapping chunks (80-word overlap) to preserve context across chunk boundaries
2. **Index** -- both a TF-IDF term matrix and a BM25 index are built in memory for each ingested document
3. **Retrieve** -- BM25 (default) or TF-IDF scores the query against all chunks; top-4 passages are returned with relevance scores normalised to [0, 100]
4. **Generate** -- retrieved passages are injected as context into a Claude API prompt; the model generates a grounded answer and is instructed to cite passage numbers

The frontend calls `/api/query` with the question, document ID, API key, and retrieval mode. A BM25/TF-IDF mode toggle in the sidebar switches retrieval algorithms without reloading the document. The `/api/compare` endpoint runs both algorithms on the same query without generation -- useful for seeing exactly where they agree and where they diverge.

Demo documents (DBS Group 2023 and Singapore Airlines FY2022/23 annual report excerpts) are auto-ingested at startup.

---

## Retrieval: TF-IDF vs BM25

Both methods are implemented from scratch using only Python standard library and `math`. No retrieval library is used.

**TF-IDF + cosine similarity:**  
Score = dot(q\_tfidf, chunk\_tfidf) / (|q| × |chunk|). Normalised to [0, 1]. Penalises repeated terms proportionally to term frequency. Blind to chunk length.

**Okapi BM25** (k1=1.5, b=0.75):  
score(q, d) = Σ\_t [ IDF(t) × tf(t,d)×(k1+1) / (tf(t,d) + k1×(1 − b + b×|d|/avgdl)) ]  
Non-linear term frequency saturation (extra occurrences give diminishing returns) and document-length normalisation. Standard default in Elasticsearch and Lucene. Generally outperforms TF-IDF on variable-length chunks.

Both are lexical -- they match exact tokens. "Earnings" does not match "net income" unless both terms appear. For financial reports where terminology is consistent this is usually fine. Dense embedding retrieval (OpenAI `text-embedding-3-small`, Cohere `embed-v3`) would improve recall on noisy or multilingual documents at the cost of an API dependency and a vector store.

---

## Quick Start

```bash
git clone https://github.com/nobaadi/earningsiq
cd earningsiq
docker compose up --build
```

Open `frontend/index.html` in your browser. The backend runs at `http://localhost:8003`. Demo documents are indexed automatically at startup.

You need an Anthropic API key to generate answers. Enter it in the sidebar -- it is sent directly to the Claude API and is never stored or logged.

### Manual

```bash
cd backend
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8003
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ingest` | Index a document. Body: `{document_id, text, title}` |
| POST | `/api/query` | Query with generation. Body: `{document_id, query, api_key, retrieval_mode}` where `retrieval_mode` is `"bm25"` (default) or `"tfidf"` |
| POST | `/api/compare` | Run both TF-IDF and BM25 on the same query, return retrieved passages and agreement rate. No API key required. Body: `{document_id, query}` |
| GET | `/api/documents` | List all indexed documents with chunk counts |
| DELETE | `/api/documents/{id}` | Remove a document from the index |
| GET | `/health` | Health check with indexed doc count and available retrieval modes |

Interactive docs at `http://localhost:8003/docs`.

### Query example

```bash
curl -X POST http://localhost:8003/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "dbs_2023",
    "query": "What was DBS net profit in 2023?",
    "api_key": "sk-ant-...",
    "retrieval_mode": "bm25"
  }'
```

### Compare retrieval example (no API key needed)

```bash
curl -X POST http://localhost:8003/api/compare \
  -H "Content-Type: application/json" \
  -d '{"document_id": "dbs_2023", "query": "dividend payout ratio"}'
```

```json
{
  "tfidf": [{"text": "...", "score": 72.1, "index": 4}, ...],
  "bm25":  [{"text": "...", "score": 100.0, "index": 4}, ...],
  "agreement": {
    "shared_chunks": 3,
    "total_retrieved": 4,
    "rate": 0.75,
    "note": "High agreement means both methods rank the same passages..."
  }
}
```

---

## Project Structure

```
earningsiq/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, startup ingest, all endpoints
│   │   └── services/rag.py      # TFIDFIndex, BM25Index, RAGPipeline
│   ├── data/
│   │   ├── dbs_2023.txt         # DBS Group 2023 annual report excerpt
│   │   └── sia_2023.txt         # Singapore Airlines FY2022/23 report excerpt
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html               # Chat interface with BM25/TF-IDF mode toggle (vanilla JS)
├── docker-compose.yml
└── README.md
```

---

## Extending to Real PDF Ingestion

The `/api/ingest` endpoint accepts any text. To ingest a real PDF:

```python
import httpx, pdfplumber

with pdfplumber.open("report.pdf") as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)

httpx.post("http://localhost:8003/api/ingest", json={
    "document_id": "my_report",
    "text": text,
    "title": "My Company Annual Report 2024",
})
```

To upgrade to dense vector retrieval, replace `BM25Index` with a class that calls an embedding API and stores vectors in a NumPy array or Postgres pgvector column. The `RAGPipeline` interface (`.ingest()`, `.query()`, `.compare_retrieval()`) stays the same.
