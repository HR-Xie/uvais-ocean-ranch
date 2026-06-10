"""FastAPI 后端 — Agent + RAG REST API + SSE 流式对话"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="UVAIS Agent API", version="2.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Lazy singletons ────────────────────────────────────────────────────

_agent = None
_rag = None

def get_agent():
    global _agent
    if _agent is None:
        from agent.react_agent import ReactAgent
        _agent = ReactAgent()
    return _agent

def get_rag():
    global _rag
    if _rag is None:
        from rag.engine import RagService
        _rag = RagService()
    return _rag

# ── Request / Response models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    history: list[dict] | None = None
    role: str = "manager"

class RAGRequest(BaseModel):
    query: str

# ── Health ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0"}

# ── Streaming Chat (SSE) ───────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    q = f"{req.role}_{req.query}" if req.role in ("manager", "worker") else req.query

    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            for chunk in get_agent().execute_stream(q, history=req.history):
                queue.put_nowait(chunk)
        except Exception as e:
            queue.put_nowait(f"\n【异常】{e}")
        finally:
            queue.put_nowait(None)

    async def _stream():
        asyncio.get_event_loop().run_in_executor(None, _run)
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield f"data: {json.dumps({'content': chunk})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")

# ── Non-streaming Chat ─────────────────────────────────────────────────

class SyncChatResponse(BaseModel):
    content: str

@app.post("/chat/sync", response_model=SyncChatResponse)
def chat_sync(req: ChatRequest):
    q = f"{req.role}_{req.query}" if req.role in ("manager", "worker") else req.query
    parts = list(get_agent().execute_stream(q, history=req.history))
    return SyncChatResponse(content="".join(parts))

# ── RAG-only ───────────────────────────────────────────────────────────

class RAGResponse(BaseModel):
    answer: str

@app.post("/rag/query", response_model=RAGResponse)
def rag_query(req: RAGRequest):
    try:
        result = get_rag().rag_summarize(req.query)
        return RAGResponse(answer=result)
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Knowledge Base ─────────────────────────────────────────────────────

@app.post("/kb/ingest")
def kb_ingest():
    try:
        n = get_rag().build_knowledge_base()
        return {"status": "ok", "nodes_created": n}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Static Frontend ────────────────────────────────────────────────────

static = Path(__file__).parent / "static"
if static.exists():
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (static / "index.html").read_text(encoding="utf-8")
    app.mount("/static", StaticFiles(directory=str(static)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
