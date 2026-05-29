# URO-CARE RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot for URO-CARE Urology & Andrology Center, Nairobi.

**Stack:** Flask · ChromaDB · OpenAI GPT-4o mini · text-embedding-3-small

---

## Architecture

```
User Browser
     │
     ▼
 Flask (app.py)
     │
     ├── GET  /           → serves the chat widget (templates/index.html)
     ├── GET  /health     → ChromaDB status check
     ├── POST /chat        → non-streaming RAG response
     └── POST /chat/stream → streaming RAG response (SSE)
          │
          ├── 1. Embed user query  ──► OpenAI text-embedding-3-small
          ├── 2. Retrieve context  ──► ChromaDB (top-5 cosine-similar chunks)
          ├── 3. Build prompt      ──► System + context + chat history
          └── 4. Stream response   ──► OpenAI GPT-4o mini (streaming)
```

---

## Project Structure

```
urocare-chatbot/
├── app.py                          # Flask backend + RAG logic
├── ingest.py                       # One-time KB ingestion into ChromaDB
├── urocare_rag_knowledge_base.md   # Knowledge base (scraped from website)
├── requirements.txt
├── README.md
├── chroma_db/                      # Auto-created by ingest.py (gitignore this)
│   └── ...
└── templates/
    └── index.html                  # Chat widget frontend
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- An **OpenAI API key** — get one at https://platform.openai.com/api-keys

### 2. Install dependencies

```bash
cd urocare-chatbot
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Or create a .env file (see note below)
```

> **Tip:** Create a `.env` file and use `python-dotenv` to load it:
> ```
> OPENAI_API_KEY=sk-...
> ```
> Then add `from dotenv import load_dotenv; load_dotenv()` at the top of `app.py`.

### 4. Ingest the knowledge base (run once)

```bash
python ingest.py
```

This will:
- Parse `urocare_rag_knowledge_base.md` into 14 sections
- Chunk each section (~400 words, 60-word overlap)
- Generate embeddings via `text-embedding-3-small`
- Persist everything in `./chroma_db/`

Expected output:
```
=== URO-CARE Knowledge Base Ingestion ===
Loading: urocare_rag_knowledge_base.md
Sections found: 14
  • SECTION 1: CLINIC OVERVIEW
  • SECTION 2: CONTACT & LOCATION
  ...
Total chunks: ~45
Generating embeddings…
  Embedded 45/45 chunks…
✅  Ingestion complete — 45 chunks stored in 'urocare_kb'
```

### 5. Start the server

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## API Reference

### `GET /health`
Returns ChromaDB status and chunk count.
```json
{ "status": "ok", "chunks_in_db": 45 }
```

### `POST /chat`
Non-streaming response.
```json
// Request
{ "message": "What are your opening hours?", "history": [] }

// Response
{
  "reply": "URO-CARE is open Monday to Friday from 9:00 AM to 5:00 PM...",
  "sources": [
    { "section": "SECTION 2: CONTACT & LOCATION", "relevance": 94.2 }
  ]
}
```

### `POST /chat/stream`
Server-Sent Events stream. Each event is `data: {...}\n\n`.

Token events:
```json
{ "token": "URO" }
{ "token": "-CARE" }
```

Done event (last):
```json
{
  "done": true,
  "sources": [{ "section": "SECTION 2: CONTACT & LOCATION", "relevance": 94.2 }]
}
```

---

## Configuration

Edit the constants at the top of `app.py`:

| Variable    | Default            | Description                          |
|-------------|--------------------|--------------------------------------|
| `CHAT_MODEL`  | `gpt-4o-mini`    | OpenAI chat model                    |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model              |
| `TOP_K`       | `5`              | Chunks retrieved per query           |
| `MAX_HISTORY` | `10`             | Conversation turns kept in memory    |

---

## Updating the Knowledge Base

1. Edit `urocare_rag_knowledge_base.md`
2. Re-run `python ingest.py` — it drops and recreates the collection
3. Restart `app.py`

---

## Deployment Tips

**Gunicorn (production):**
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

**Docker** (create `Dockerfile`):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python ingest.py
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
```

**Railway / Render / Fly.io:** Set `OPENAI_API_KEY` as an environment variable in the dashboard.

---

## Cost Estimate

| Operation            | Model                     | Approx. cost         |
|----------------------|---------------------------|----------------------|
| Ingestion (one-time) | text-embedding-3-small    | ~$0.001 total        |
| Per user message     | text-embedding-3-small    | ~$0.00002            |
| Per AI response      | gpt-4o-mini               | ~$0.0003–0.001       |

Roughly **$0.001 per conversation** at typical usage.

---

## License

Built for URO-CARE Urology & Andrology Center, Nairobi.
