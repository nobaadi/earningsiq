import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.rag import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EarningsIQ API",
    description="RAG pipeline for financial report Q&A: TF-IDF retrieval + Claude generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()

_DATA_DIR = Path(__file__).parent.parent / "data"
_DEMO_DOCS = [
    ("dbs_2023", "DBS Annual Report 2023", "dbs_2023.txt"),
    ("sia_2023", "Singapore Airlines FY2022/23 Annual Report", "sia_2023.txt"),
]


@app.on_event("startup")
async def startup():
    for doc_id, title, filename in _DEMO_DOCS:
        path = _DATA_DIR / filename
        if path.exists():
            text = path.read_text(encoding="utf-8")
            n = rag.ingest(doc_id, text, title)
            logger.info("Auto-ingested '%s': %d chunks", title, n)
        else:
            logger.warning("Demo document not found: %s", path)


class QueryRequest(BaseModel):
    query: str
    api_key: str
    document_id: str


class IngestRequest(BaseModel):
    document_id: str
    text: str
    title: str


@app.post("/api/ingest")
async def ingest_document(req: IngestRequest):
    chunk_count = rag.ingest(req.document_id, req.text, req.title)
    return {"document_id": req.document_id, "chunks": chunk_count, "status": "indexed"}


@app.post("/api/query")
async def query_document(req: QueryRequest):
    if not req.api_key.startswith("sk-ant-"):
        raise HTTPException(status_code=400, detail="Invalid Anthropic API key format (must start with sk-ant-)")
    result = await rag.query(req.document_id, req.query, req.api_key)
    return result


@app.get("/api/documents")
async def list_documents():
    return {"documents": rag.list_documents()}


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    rag.delete(document_id)
    return {"deleted": document_id}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "earningsiq", "indexed_docs": len(rag.list_documents())}
