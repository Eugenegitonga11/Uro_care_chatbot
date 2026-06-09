"""
app.py  —  URO-CARE RAG Chatbot Backend
Flask + JSON Vector Store + NVIDIA API
"""

import os
import json
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
VECTOR_STORE    = "./vector_store.json"
EMBED_MODEL     = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL      = "meta/llama-3.1-8b-instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
TOP_K           = 5
MAX_HISTORY     = 10

# Keywords that trigger the booking form
BOOKING_KEYWORDS = [
    "book", "appointment", "schedule", "reserve", "visit",
    "come in", "see a doctor", "consult", "consultation",
    "how do i book", "want to book", "make an appointment"
]

SYSTEM_PROMPT = """You are a warm, professional and knowledgeable patient care assistant for URO-CARE Urology & Andrology Center — Nairobi's premier specialist clinic.

You answer questions ONLY using the context provided below from the URO-CARE knowledge base. If the answer is not in the context, say you don't have that specific information and direct the patient to call +254 112 288 709 or WhatsApp for personalised help.

Core rules:
1. NEVER provide a medical diagnosis or prescribe treatment.
2. Keep replies concise, warm and human — 2-5 sentences for simple questions, slightly longer for complex ones.
3. Always end with a helpful next step (book an appointment, call us, WhatsApp us).
4. Use plain language — no jargon unless explaining a term the patient asked about.
5. Be reassuring and empathetic, especially for sensitive topics (ED, infertility, STIs).
6. For emergencies or urgent symptoms, always advise calling +254 112 288 709 immediately.
7. When a patient wants to book an appointment, let them know you will show them a quick form to get started.

Contact info to always have ready:
- Phone / WhatsApp: +254 112 288 709
- Email: info@urocare.co.ke
- Location: 4th Floor, PMC Building, 3rd Parklands Avenue, Nairobi
- Hours: Mon-Fri 9 AM-5 PM | Sat 10 AM-3 PM
- No referral needed. Same-week appointments available. 100% confidential.

--- KNOWLEDGE BASE CONTEXT ---
{context}
--- END CONTEXT ---"""

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

_openai_client = None
_vector_store  = None


def get_openai():
    global _openai_client
    if _openai_client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set in .env file.")
        _openai_client = OpenAI(api_key=key, base_url=NVIDIA_BASE_URL)
        logger.info("NVIDIA client ready.")
    return _openai_client


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        if not os.path.exists(VECTOR_STORE):
            raise RuntimeError("vector_store.json not found. Run: python ingest.py")
        with open(VECTOR_STORE, "r", encoding="utf-8") as f:
            _vector_store = json.load(f)
        logger.info(f"Vector store loaded — {len(_vector_store)} chunks.")
    return _vector_store


def cosine_similarity(a, b):
    try:
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


def embed_query(text):
    resp = get_openai().embeddings.create(
        model=EMBED_MODEL,
        input=[text],
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"}
    )
    return resp.data[0].embedding


def search(query, k=TOP_K):
    try:
        q_emb = embed_query(query)
        store = get_vector_store()
        if not store:
            return []
        scored = []
        for item in store:
            try:
                score = cosine_similarity(q_emb, item.get("embedding", []))
                scored.append((score, item))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:min(k, len(scored))]
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


def retrieve_context(query):
    results = search(query)
    if not results:
        return "No relevant context found."
    parts = []
    for score, item in results:
        relevance = round(score * 100, 1)
        section   = item.get("section", "General")
        text      = item.get("text", "")
        parts.append(f"[Section: {section} | Relevance: {relevance}%]\n{text}")
    return "\n\n---\n\n".join(parts)


def get_sources(query):
    results = search(query)
    seen, sources = set(), []
    for score, item in results:
        section = item.get("section", "General")
        if score > 0.3 and section not in seen:
            seen.add(section)
            sources.append({"section": section, "relevance": round(score * 100, 1)})
    return sources


def build_messages(history, context, user_message):
    system   = SYSTEM_PROMPT.format(context=context)
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-(MAX_HISTORY * 2):])
    messages.append({"role": "user", "content": user_message})
    return messages


def is_booking_intent(message):
    """Check if the message contains booking intent."""
    msg_lower = message.lower()
    return any(keyword in msg_lower for keyword in BOOKING_KEYWORDS)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        store = get_vector_store()
        return jsonify({
            "status":       "ok",
            "chunks_in_db": len(store),
            "chat_model":   CHAT_MODEL,
            "embed_model":  EMBED_MODEL,
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data    = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        history = data.get("history") or []

        if not message:
            return jsonify({"error": "No message provided"}), 400

        # Detect booking intent
        show_booking_form = is_booking_intent(message)

        context  = retrieve_context(message)
        messages = build_messages(history, context, message)

        resp  = get_openai().chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=600,
        )
        reply   = resp.choices[0].message.content
        sources = get_sources(message)

        return jsonify({
            "reply":             reply,
            "sources":           sources,
            "show_booking_form": show_booking_form
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "reply":             "I'm experiencing a technical issue. Please call +254 112 288 709!",
            "sources":           [],
            "show_booking_form": False
        }), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in .env file.")
        input("Press Enter to close...")
        exit(1)

    if not os.path.exists(VECTOR_STORE):
        logger.error("vector_store.json not found. Run: python ingest.py first.")
        input("Press Enter to close...")
        exit(1)

    logger.info("=== URO-CARE RAG Chatbot ===")
    logger.info(f"Chat model:   {CHAT_MODEL}")
    logger.info(f"Embed model:  {EMBED_MODEL}")
    logger.info(f"Vector store: {os.path.abspath(VECTOR_STORE)}")
    logger.info("Open browser at: http://127.0.0.1:5000")
    logger.info("Press CTRL+C to stop")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )