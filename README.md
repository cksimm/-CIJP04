# 📘 Local RAG Chatbot (FastAPI + FAISS + FastEmbed + Ollama)

A local retrieval-augmented chatbot that indexes PDFs and answers questions using a local LLM via [Ollama](https://ollama.com).
This version uses **[FastEmbed](https://github.com/qdrant/fastembed)** instead of PyTorch for compatibility with Python 3.13.

---

## 🚀 Requirements

- macOS / Linux / WSL2
- Python **3.12+** (tested on 3.13)
- [Ollama](https://ollama.com/download) installed
- At least one LLM pulled (e.g. `qwen2.5:7b-instruct` or `qwen2.5:3b-instruct`)

---

## 📦 Installation

1. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Environment Variables

You can set these in your shell or in a `.env` file:

```bash
# Ollama host (use your port if not 11434)
export OLLAMA_HOST=127.0.0.1:11500

# Chat model served by Ollama
export MODEL_NAME=qwen2.5:7b-instruct

# Embedding model for retrieval
export EMBED_MODEL=BAAI/bge-small-en-v1.5
```

---

## 🖥️ Running Ollama

You need **two terminal windows**:

1. **Terminal 1 → start Ollama server**
   ```bash
   ollama serve
   ```
   Leave this running. It will keep the model API alive on your chosen port (default 11434 unless overridden).

   Pull at least one model (in any terminal):
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```
   *(Use `qwen2.5:3b-instruct` if you want something lighter/faster.)*

2. **Terminal 2 → run the chatbot**
   - Make sure your venv is active.
   - Build the index (only after you put PDFs in `./data/`):
     ```bash
     python ingest.py
     ```
   - Start the API:
     ```bash
     export OLLAMA_HOST=127.0.0.1:11500
     python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
     ```

   Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📚 Indexing Your PDFs

1. Place your documents in the `./data/` folder.
2. Run:
   ```bash
   python ingest.py
   ```
3. This generates:
   - `./chroma_db/index.faiss` (vector index)
   - `./chroma_db/meta.jsonl` (chunk metadata)

---

## 💬 Using the Chatbot

- Open http://localhost:8000
- Type a question about your PDFs, e.g.:
  *“What topics are in the NBHWC Content Outline?”*

The bot retrieves relevant chunks and responds with an answer + citations.

---

## 🔍 Troubleshooting

- **Vectors = 0** in `/health` → run `python ingest.py` again.
- **Ollama 404 errors** → ensure you exported the correct `OLLAMA_HOST` in the **same terminal** where you run `uvicorn`.
- **No answer in UI but works via curl** → check browser console (⌥⌘I on Mac) for JavaScript errors.
