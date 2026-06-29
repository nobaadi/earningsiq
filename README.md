# EarningsIQ: Financial Report Q&A with RAG

A retrieval-augmented generation (RAG) system for querying annual reports. You select a document, ask a question in plain English, and get a grounded answer that cites the specific passages it was drawn from.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Claude](https://img.shields.io/badge/Claude-Haiku_4.5-orange)](https://anthropic.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)

---

## What It Does

The backend implements a four-step RAG pipeline:

1. **Ingest** -- document text is split into 500-word overlapping chunks (80-word overlap) to preserve context across boundaries
2. **Index** -- a TF-IDF term matrix is built in memory for each document; no external vector database required
3. **Retrieve** -- cosine similarity scores the query against all chunks; top-4 passages are returned with relevance percentages
4. **Generate** -- retrieved passages are injected as context into a Claude API prompt; the model generates a grounded answer

The frontend calls `/api/query` with the question, document ID, and API key. The backend handles all retrieval and generation, then returns the answer plus the retrieved passages and their scores.

Demo documents (DBS Group 2023 and Singapore Airlines FY2022/23 annual report excerpts) are auto-ingested when the backend starts.

---

## Why TF-IDF Instead of a Vector Database

This is an intentional design choice worth explaining in an interview:

**What TF-IDF gives you:** zero infrastructure, deterministic retrieval, fully explainable scores, sub-millisecond latency, no embedding API costs. For financial reports where terminology is consistent (revenue, net profit, margin appear verbatim in both the question and the document), lexical matching is sufficient.

**What you lose:** semantic recall. TF-IDF misses synonyms and paraphrases. "Earnings" doesn't match "net income" unless both terms appear. A dense embedding model (OpenAI `text-embedding-3-small`, Cohere `embed-v3`) would handle this, at the cost of a vector DB (Pinecone, Weaviate, Postgres pgvector) and embedding API calls.

**The right tradeoff depends on the use case.** For a portfolio demo and for interviews, TF-IDF is better because the mechanics are on one page and you can explain every retrieval decision. For production on noisy or multilingual documents, dense retrieval wins.

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
| POST | `/api/query` | Query an indexed document. Body: `{document_id, query, api_key}` |
| GET | `/api/documents` | List all indexed documents with chunk counts |
| DELETE | `/api/documents/{id}` | Remove a document from the index |
| GET | `/health` | Health check with indexed document count |

Interactive docs at `http://localhost:8003/docs`.

### Example query

```bash
curl -X POST http://localhost:8003/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "dbs_2023",
    "query": "What was DBS net profit in 2023?",
    "api_key": "sk-ant-..."
  }'
```

Response:
```json
{
  "answer": "DBS Group reported record net profit of SGD 10.3 billion for the full year 2023...",
  "retrieved": [
    {"text": "DBS Group reported record net profit...", "score": 84.2, "index": 0},
    ...
  ],
  "model": "claude-haiku-4-5-20251001"
}
```

---

## Project Structure

```
earningsiq/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, auto-ingest startup, endpoints
│   │   └── services/rag.py      # TF-IDF index, cosine retrieval, Claude generation
│   ├── data/
│   │   ├── dbs_2023.txt         # DBS Group 2023 annual report excerpt
│   │   └── sia_2023.txt         # Singapore Airlines FY2022/23 report excerpt
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html               # Chat interface (vanilla JS, no build step)
├── docker-compose.yml
└── README.md
```

---

## Extending to Real PDF Ingestion

The `/api/ingest` endpoint accepts any text. To ingest a real PDF:

```python
import httpx
import pdfplumber

with pdfplumber.open("report.pdf") as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)

httpx.post("http://localhost:8003/api/ingest", json={
    "document_id": "my_report",
    "text": text,
    "title": "My Company Annual Report 2024",
})
```

To upgrade to dense vector retrieval, replace `TFIDFIndex` with a class that calls an embedding API and stores vectors in a NumPy array or Postgres pgvector column. The `RAGPipeline` interface (`.ingest()`, `.query()`) stays the same -- the generation step is independent of the retrieval method.
