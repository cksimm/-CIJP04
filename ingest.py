#!/usr/bin/env python3
"""
Build a FAISS index from PDFs in ./data using FastEmbed (no PyTorch).
Outputs:
  - ./chroma_db/index.faiss
  - ./chroma_db/meta.jsonl         (one JSON per chunk: {"file","page","chunk","text"})
Env (optional):
  - DATA_DIR   (default: ./data)
  - CHROMA_DIR (default: ./chroma_db)
  - EMBED_MODEL (default: BAAI/bge-small-en-v1.5)
"""

from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss

try:
    from pypdf import PdfReader
except Exception as e:
    raise SystemExit("Missing dependency 'pypdf'. Install with: pip install pypdf") from e

try:
    from fastembed import TextEmbedding
except Exception as e:
    raise SystemExit("Missing dependency 'fastembed'. Install with: pip install fastembed") from e

try:
    from tqdm import tqdm
except Exception:
    # fallback if tqdm not installed
    def tqdm(x, **kwargs): return x


DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma_db"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Model dims for common FastEmbed models (kept to pre-size arrays if desired)
# bge-small-en-v1.5 -> 384, bge-base-en-v1.5 -> 768, bge-large-en-v1.5 -> 1024
KNOWN_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
}

CHUNK_SIZE = 800       # characters
CHUNK_OVERLAP = 120    # characters


def read_pdfs(datadir: Path) -> List[Tuple[str, int, str]]:
    """Return list of (filename, page_number_1based, page_text)."""
    pages = []
    for pdf in sorted(datadir.glob("*.pdf")):
        reader = PdfReader(str(pdf))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((pdf.name, i + 1, text))
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_corpus(pages: List[Tuple[str, int, str]]) -> Tuple[List[str], List[Dict]]:
    texts: List[str] = []
    metas: List[Dict] = []
    for fname, page_num, page_text in pages:
        for ci, chunk in enumerate(chunk_text(page_text)):
            texts.append(chunk)
            metas.append({"file": fname, "page": page_num, "chunk": ci})
    return texts, metas


def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    embedder = TextEmbedding(model_name=model_name)
    # fastembed returns an iterator; materialize to array
    vecs = np.array(list(embedder.embed(texts)), dtype="float32")
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(vecs)
    return vecs


def build_index(vecs: np.ndarray) -> faiss.Index:
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner-product (cosine if normalized)
    index.add(vecs)
    return index


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Data folder not found: {DATA_DIR.resolve()} (put PDFs there)")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading PDFs from: {DATA_DIR}")
    pages = read_pdfs(DATA_DIR)
    if not pages:
        raise SystemExit(f"No PDFs with extractable text in {DATA_DIR}")

    print("Chunking pages...")
    texts, metas = build_corpus(pages)
    print(f"Total chunks: {len(texts)}")

    dim_hint = KNOWN_DIMS.get(EMBED_MODEL, None)
    if dim_hint:
        print(f"Embedding with {EMBED_MODEL} (dim={dim_hint})")
    else:
        print(f"Embedding with {EMBED_MODEL}")

    vecs = embed_texts(tqdm(texts, desc="Embedding"), EMBED_MODEL)

    index = build_index(vecs)
    faiss.write_index(index, str(CHROMA_DIR / "index.faiss"))
    with open(CHROMA_DIR / "meta.jsonl", "w", encoding="utf-8") as f:
        for text, meta in zip(texts, metas):
            meta = dict(meta)
            meta["text"] = text
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {CHROMA_DIR / 'index.faiss'} and {CHROMA_DIR / 'meta.jsonl'}")


if __name__ == "__main__":
    main()
