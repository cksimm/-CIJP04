#!/usr/bin/env python3
"""
FastAPI RAG server using FAISS + FastEmbed and a local LLM via Ollama.

Endpoints:
  - GET  /              -> serves index.html if present; else returns a minimal page
  - POST /chat          -> {message:str, k:int?, temperature:float?} -> RAG answer + sources
  - GET  /health        -> {"status":"ok"}

Env (optional):
  - CHROMA_DIR   (default: ./chroma_db)
  - EMBED_MODEL  (default: BAAI/bge-small-en-v1.5)
  - OLLAMA_HOST  (e.g. 127.0.0.1:11500)
  - OLLAMA_BASE  (e.g. http://127.0.0.1:11500)
  - MODEL_NAME   (default: qwen2.5:7b-instruct)
"""

from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
import requests

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from fastembed import TextEmbedding
except Exception as e:
    raise SystemExit("Missing dependency 'fastembed'. Install with: pip install fastembed") from e


# -------------------------
# Config
# -------------------------
ROOT = Path(__file__).parent
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b-instruct")

def _resolve_ollama_base() -> str:
    if os.getenv("OLLAMA_BASE"):
        return os.getenv("OLLAMA_BASE").rstrip("/")
    host = os.getenv("OLLAMA_HOST")
    if host:
        return f"http://{host}"
    return "http://127.0.0.1:11434"

OLLAMA_BASE = _resolve_ollama_base()
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"

MAX_CHARS_PER_CHUNK = 800
MAX_TOTAL_CONTEXT_CHARS = 4000
DEFAULT_TOP_K = 3

app = FastAPI(title="Local RAG Chat (FAISS + FastEmbed + Ollama)")

static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

index_html_path = ROOT / "index.html"

class ChatRequest(BaseModel):
    message: str
    k: Optional[int] = DEFAULT_TOP_K
    temperature: Optional[float] = 0.2

faiss_index: Optional[faiss.Index] = None
meta_rows: List[Dict[str, Any]] = []
embedder: Optional[TextEmbedding] = None

# very simple in-memory user info
user_info: Dict[str, Any] = {}

# -------------------------
# Utilities
# -------------------------
def load_index_and_meta() -> None:
    global faiss_index, meta_rows
    index_path = CHROMA_DIR / "index.faiss"
    meta_path = CHROMA_DIR / "meta.jsonl"
    if not index_path.exists() or not meta_path.exists():
        raise RuntimeError(
            f"Vector store not found in {CHROMA_DIR}. Run 'python ingest.py' after putting PDFs in ./data."
        )
    faiss_index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_rows = [json.loads(line) for line in f if line.strip()]
    if len(meta_rows) != faiss_index.ntotal:
        print("⚠️ meta.jsonl rows != FAISS ntotal. Did you replace one but not the other?")


def get_embedder() -> TextEmbedding:
    global embedder
    if embedder is None:
        embedder = TextEmbedding(model_name=EMBED_MODEL)
    return embedder


def embed_query(q: str) -> np.ndarray:
    v = np.array(list(get_embedder().embed([q]))[0], dtype="float32")
    v = v.reshape(1, -1)
    faiss.normalize_L2(v)
    return v


def search_similar(qvec: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
    assert faiss_index is not None
    D, I = faiss_index.search(qvec, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        m = meta_rows[idx] if 0 <= idx < len(meta_rows) else {"file": "unknown", "page": None, "chunk": None, "text": ""}
        results.append({"score": float(score), "meta": m})
    return results


def build_prompt(user_msg: str, context_chunks, name: Optional[str] = None):
    def trim(s: str) -> str:
        return (s or "")[:MAX_CHARS_PER_CHUNK]

    blocks, total = [], 0
    for i, r in enumerate(context_chunks):
        txt = trim(r["meta"].get("text", ""))
        block = f"[{i+1}] (file: {r['meta'].get('file')}, page: {r['meta'].get('page')})\n{txt}"
        if total + len(block) > MAX_TOTAL_CONTEXT_CHARS:
            break
        blocks.append(block)
        total += len(block)

    context_text = "\n\n".join(blocks)

    if name:
        persona_line = f"You are a helpful assistant. The user's name is {name}. Be friendly and concise. "
    else:
        persona_line = "You are a helpful assistant. Be friendly and concise. "

    system = (
        persona_line +
        "Answer using ONLY the provided context when relevant. "
        "If the answer isn't in the context, say you don't know. "
        "Cite sources by their [number] at the end of the relevant sentence."
    )

    user = (
        f"USER QUESTION:\n{user_msg}\n\n"
        f"CONTEXT (numbered):\n{context_text}\n\n"
        "When you reference context, include [1], [2], etc. Provide a concise answer."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

def call_ollama(messages, temperature: float = 0.2) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 256,
        },
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "message" in data and isinstance(data["message"], dict):
        return data["message"].get("content", "")
    return data.get("response", "")


# -------------------------
# Helper: detect intro (name/age)
# -------------------------
def extract_name_age(text: str):
    text_low = text.lower()
    name = None
    age = None

    # name pattern
    m = re.search(r"my name is\s+([a-zA-Z]+)", text_low)
    if m:
        name = m.group(1).strip().title()

    # age pattern: "i am 21", "i am 21 years old"
    m2 = re.search(r"i am\s+(\d{1,2})\s*(years old|year old|yo)?", text_low)
    if m2:
        age = m2.group(1)

    return name, age


# -------------------------
# Routes
# -------------------------
@app.on_event("startup")
def _startup():
    print("🔧 Loading FAISS index + metadata...")
    load_index_and_meta()
    print(f"✅ Loaded {faiss_index.ntotal if faiss_index else 0} vectors from {CHROMA_DIR}")
    _ = get_embedder()
    print(f"🧠 Embedder ready: {EMBED_MODEL}")
    print(f"🤖 Using Ollama at {OLLAMA_BASE} (model: {MODEL_NAME})")


@app.get("/health")
def health():
    return {"status": "ok", "vectors": faiss_index.ntotal if faiss_index else 0}


@app.post("/chat")
def chat(req: ChatRequest):
    message = (req.message or "").strip()
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    lower_msg = message.lower()

    # 1) GREETING MODE
    if lower_msg in ("hi", "hello", "hey", "hi!", "hello!", "hey!"):
        greet_messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly health coach chatbot. "
                    "When the user greets you, respond warmly and ask for their name and age "
                    "so you can personalise future advice. Keep it short. Do NOT reference PDFs."
                ),
            },
            {"role": "user", "content": message},
        ]
        try:
            answer = call_ollama(greet_messages, temperature=float(req.temperature or 0.2))
        except requests.RequestException as e:
            return JSONResponse({"error": f"Ollama call failed: {e}", "ollama_base": OLLAMA_BASE}, status_code=502)
        return {"answer": answer, "sources": []}

    # 2) INTRO MODE (user gives name/age) → we acknowledge, don't RAG
    name, age = extract_name_age(message)
    if name or age:
        if name:
            user_info["name"] = name
        if age:
            user_info["age"] = age

        intro_messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly health coach chatbot. The user has just shared their personal details. "
                    "Acknowledge their name and/or age, say it's helpful for tailoring health guidance, "
                    "and ask what they want to work on (e.g. weight, sleep, stress, chronic disease). "
                    "Do NOT mention missing context or PDFs. Keep it warm and encouraging."
                ),
            },
            {"role": "user", "content": message},
        ]
        try:
            answer = call_ollama(intro_messages, temperature=float(req.temperature or 0.2))
        except requests.RequestException as e:
            return JSONResponse({"error": f"Ollama call failed: {e}", "ollama_base": OLLAMA_BASE}, status_code=502)
        return {"answer": answer, "sources": []}

    # 3) NORMAL RAG MODE
    qvec = embed_query(message)
    hits = search_similar(qvec, top_k=max(1, int(req.k or 4)))
    current_name = user_info.get("name")
    msgs = build_prompt(message, hits, name=current_name)

    try:
        answer = call_ollama(msgs, temperature=float(req.temperature or 0.2))
    except requests.RequestException as e:
        return JSONResponse({"error": f"Ollama call failed: {e}", "ollama_base": OLLAMA_BASE}, status_code=502)

    sources = [
        {
            "file": h["meta"].get("file"),
            "page": h["meta"].get("page"),
            "chunk": h["meta"].get("chunk"),
            "score": round(h["score"], 4),
        }
        for h in hits
    ]
    return {"answer": answer, "sources": sources}


@app.get("/", response_class=HTMLResponse)
def root():
    if index_html_path.exists():
        return FileResponse(str(index_html_path))
    return HTMLResponse(
        """
        <!doctype html>
        <meta charset="utf-8"/>
        <title>Local RAG Chat</title>
        <style>
          body{font-family: ui-sans-serif,system-ui,Arial;margin:2rem;max-width:800px}
          textarea{width:100%;height:120px}
          .msg{white-space:pre-wrap;margin:1rem 0;padding:0.75rem;background:#f6f6f6;border-radius:8px}
          .sources{font-size:0.9em;color:#444}
          button{padding:0.6rem 1rem;border-radius:8px;border:1px solid #ccc;background:#fff;cursor:pointer}
          button:disabled{opacity:.6;cursor:not-allowed}
        </style>
        <h1>Local RAG Chat</h1>
        <p>Ask a question about your indexed PDFs.</p>
        <textarea id="q" placeholder="Type your question..."></textarea><br/>
        <label>Top-K: <input id="k" type="number" value="4" min="1" max="10" style="width:4rem"/></label>
        <label style="margin-left:1rem;">Temperature: <input id="t" type="number" value="0.2" step="0.1" style="width:5rem"/></label>
        <div style="margin-top:1rem;">
          <button id="ask">Ask</button>
        </div>
        <div id="out"></div>
        <script>
          const askBtn = document.getElementById('ask');
          askBtn.onclick = async () => {
            askBtn.disabled = true;
            const message = document.getElementById('q').value;
            const k = parseInt(document.getElementById('k').value || '4', 10);
            const temperature = parseFloat(document.getElementById('t').value || '0.2');
            const res = await fetch('/chat', {
              method:'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify({message, k, temperature})
            });
            const data = await res.json();
            askBtn.disabled = false;
            const out = document.getElementById('out');
            if (data.error) {
              out.innerHTML = '<div class="msg">Error: '+ data.error +'</div>';
              return;
            }
            const src = (data.sources||[]).map((s,i)=>`[${i+1}] ${s.file} p.${s.page} (score ${s.score})`).join('<br/>');
            out.innerHTML = '<div class="msg">'+ (data.answer||'(no answer)') +'</div>' +
                            '<div class="sources"><b>Sources</b><br/>' + src + '</div>';
          };
        </script>
        """.strip()
    )