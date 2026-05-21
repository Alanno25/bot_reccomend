"""
Bangkok Bless Asset — FastAPI Web Server
POST /chat   → RAG chatbot (stateless: frontend keeps history)
GET  /health → health check
GET  /       → API info
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from house_rec import init_pipeline, rag_chat


# ──────────────────────────────────────────
# Schema
# ──────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []      # frontend stores and sends history each request


class ListingItem(BaseModel):
    name: str | None = None
    type: str | None = None
    district: str | None = None
    province: str | None = None
    price_thb: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    distance_km: float | None = None
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    mode: str                     # "location" or "semantic"
    sources: list[ListingItem]
    history: list[dict]           # updated history — send back next request


# ──────────────────────────────────────────
# Lifespan — load model once at startup
# ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = init_pipeline()
    yield


# ──────────────────────────────────────────
# App
# ──────────────────────────────────────────
app = FastAPI(
    title="Bangkok Bless Asset RAG Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Bangkok Bless Asset RAG Chatbot",
        "endpoints": {"chat": "POST /chat", "health": "GET /health"},
    }


@app.get("/health")
def health():
    pipeline = getattr(app.state, "pipeline", None)
    return {"status": "ok" if pipeline else "loading"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    p = app.state.pipeline
    try:
        answer, sources, mode = rag_chat(
            req.query,
            req.history,
            p["embed_model"],
            p["idx"],
            p["docs"],
            p["llm"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build updated history (stateless: send back to frontend)
    updated_history = req.history + [
        {"role": "user",      "content": req.query},
        {"role": "assistant", "content": answer},
    ]

    listing_items = [
        ListingItem(
            name=doc.get("name"),
            type=doc.get("type"),
            district=doc.get("district"),
            province=doc.get("province"),
            price_thb=doc.get("price_thb"),
            latitude=doc.get("latitude"),
            longitude=doc.get("longitude"),
            distance_km=round(val, 2) if mode == "location" else None,
            url=doc.get("url"),
        )
        for doc, val in sources
    ]

    return ChatResponse(
        answer=answer,
        mode=mode,
        sources=listing_items,
        history=updated_history,
    )
