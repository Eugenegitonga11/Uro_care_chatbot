"""
app.py — URO-CARE RAG Chatbot Backend

Flask + JSON Vector Store + NVIDIA API
"""

import os
import json
import logging
import time

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv


# =============================================================================
# ENVIRONMENT
# =============================================================================

load_dotenv()


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

VECTOR_STORE = "./vector_store.json"

# IMPORTANT:
# This MUST be the same embedding model used in ingest.py.
EMBED_MODEL = "nvidia/nemotron-3-embed-1b"

# Current NVIDIA chat model
CHAT_MODEL = "nvidia/nemotron-3-super-120b-a12b"

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Number of chunks retrieved from the vector store.
TOP_K = 5

# Number of previous conversation messages to send to the model.
MAX_HISTORY = 5

# Maximum time allowed for an NVIDIA API request.
API_TIMEOUT = 20.0


# =============================================================================
# BOOKING KEYWORDS
# =============================================================================

BOOKING_KEYWORDS = [
    "book",
    "appointment",
    "schedule",
    "reserve",
    "visit",
    "come in",
    "see a doctor",
    "consult",
    "consultation",
    "how do i book",
    "want to book",
    "make an appointment"
]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """
You are a warm, professional and knowledgeable patient care assistant
for URO-CARE Urology & Andrology Center in Nairobi.

You answer questions ONLY using the knowledge-base context provided below.

If the answer is not contained in the context, say that you don't have
that specific information and direct the patient to call or WhatsApp
URO-CARE for personalised assistance.

CORE RULES:

1. NEVER provide a medical diagnosis.

2. NEVER prescribe medication or treatment.

3. Keep replies concise, warm and human.

4. For simple questions, normally respond in 2-5 sentences.

5. Use plain language and avoid unnecessary medical jargon.

6. Be reassuring and empathetic, especially for sensitive subjects
   such as erectile dysfunction, infertility and STIs.

7. For emergencies or urgent symptoms, advise the patient to seek
   urgent medical attention and contact URO-CARE.

8. Always provide a helpful next step when appropriate.

CONTACT INFORMATION:

Phone / WhatsApp:
+254 112 268 709

Email:
info@urocare.co.ke

Location:
4th Floor, PMC Building,
3rd Parklands Avenue, Nairobi

Hours:
Monday-Friday: 9 AM-5 PM
Saturday: 10 AM-3 PM

No referral needed.
Same-week appointments may be available.
Patient confidentiality is respected.

--- KNOWLEDGE BASE CONTEXT ---

{context}

--- END KNOWLEDGE BASE CONTEXT ---
"""


# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)

CORS(app)


# =============================================================================
# GLOBAL OBJECTS
# =============================================================================

_openai_client = None
_vector_store = None


# =============================================================================
# NVIDIA CLIENT
# =============================================================================

def get_openai():
    """
    Create and reuse the NVIDIA/OpenAI-compatible client.
    """

    global _openai_client

    if _openai_client is None:

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set in .env file."
            )

        _openai_client = OpenAI(
            api_key=api_key,
            base_url=NVIDIA_BASE_URL,
            timeout=API_TIMEOUT,
            max_retries=0
        )

        logger.info("NVIDIA client ready.")

    return _openai_client


# =============================================================================
# VECTOR STORE
# =============================================================================

def get_vector_store():
    """
    Load vector_store.json once and keep it in memory.
    """

    global _vector_store

    if _vector_store is None:

        if not os.path.exists(VECTOR_STORE):
            raise RuntimeError(
                "vector_store.json not found. "
                "Run: python ingest.py"
            )

        with open(
            VECTOR_STORE,
            "r",
            encoding="utf-8"
        ) as f:

            _vector_store = json.load(f)

        logger.info(
            f"Vector store loaded — "
            f"{len(_vector_store)} chunks."
        )

    return _vector_store


# =============================================================================
# COSINE SIMILARITY
# =============================================================================

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    """

    try:

        a = np.array(
            a,
            dtype=np.float32
        )

        b = np.array(
            b,
            dtype=np.float32
        )

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(
            np.dot(a, b)
            /
            (norm_a * norm_b)
        )

    except Exception as e:

        logger.warning(
            f"Cosine similarity error: {e}"
        )

        return 0.0


# =============================================================================
# QUERY EMBEDDING
# =============================================================================

def embed_query(text):
    """
    Convert the user's question into an embedding.

    Query embeddings use input_type='query'.

    The knowledge-base documents should have been embedded using
    input_type='passage' in ingest.py.
    """

    logger.info("Generating query embedding...")

    start_time = time.perf_counter()

    response = get_openai().embeddings.create(
        model=EMBED_MODEL,
        input=[text],
        encoding_format="float",
        extra_body={
            "input_type": "query",
            "truncate": "END"
        }
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        f"Query embedding generated in "
        f"{elapsed:.2f}s."
    )

    return response.data[0].embedding


# =============================================================================
# VECTOR SEARCH
# =============================================================================

def retrieve_results(query, k=TOP_K):
    """
    Perform vector similarity search.

    This function performs ONE embedding request per user message.

    The results are reused for:
    - RAG context
    - Source information
    """

    try:

        # ---------------------------------------------------------
        # Generate query embedding
        # ---------------------------------------------------------

        query_embedding = embed_query(query)

        # ---------------------------------------------------------
        # Load vector store
        # ---------------------------------------------------------

        store = get_vector_store()

        if not store:
            logger.warning(
                "Vector store is empty."
            )
            return []

        # ---------------------------------------------------------
        # Calculate similarity
        # ---------------------------------------------------------

        scored = []

        for item in store:

            embedding = item.get(
                "embedding",
                []
            )

            score = cosine_similarity(
                query_embedding,
                embedding
            )

            scored.append(
                (score, item)
            )

        # ---------------------------------------------------------
        # Highest similarity first
        # ---------------------------------------------------------

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        results = scored[
            :min(k, len(scored))
        ]

        logger.info(
            f"Vector search complete — "
            f"{len(results)} results."
        )

        # Log the scores so we can see retrieval quality.
        for index, (score, item) in enumerate(
            results,
            start=1
        ):

            logger.info(
                f"Result {index}: "
                f"{item.get('section', 'General')} "
                f"| score={score:.4f}"
            )

        return results

    except Exception as e:

        logger.error(
            f"Search error: {e}",
            exc_info=True
        )

        return []


# =============================================================================
# BUILD RAG CONTEXT
# =============================================================================

def build_context(results):
    """
    Convert retrieved chunks into context for the LLM.
    """

    if not results:

        return "No relevant context found."

    parts = []

    for score, item in results:

        relevance = round(
            score * 100,
            1
        )

        section = item.get(
            "section",
            "General"
        )

        text = item.get(
            "text",
            ""
        )

        parts.append(
            f"[Section: {section} | "
            f"Relevance: {relevance}%]\n"
            f"{text}"
        )

    context = "\n\n---\n\n".join(parts)

    logger.info(
        f"RAG context size: "
        f"{len(context)} characters."
    )

    return context


# =============================================================================
# BUILD SOURCES
# =============================================================================

def build_sources(results):
    """
    Build source information from the SAME
    vector-search results.

    This avoids another embedding API request.
    """

    seen = set()
    sources = []

    for score, item in results:

        section = item.get(
            "section",
            "General"
        )

        # Ignore weak matches.
        if score <= 0.3:
            continue

        if section in seen:
            continue

        seen.add(section)

        sources.append(
            {
                "section": section,
                "relevance": round(
                    score * 100,
                    1
                )
            }
        )

    return sources


# =============================================================================
# BUILD CHAT MESSAGES
# =============================================================================

def build_messages(
    history,
    context,
    user_message
):
    """
    Build the final messages sent to the chat model.
    """

    system_message = SYSTEM_PROMPT.format(
        context=context
    )

    messages = [
        {
            "role": "system",
            "content": system_message
        }
    ]

    # ---------------------------------------------------------
    # Conversation history
    # ---------------------------------------------------------

    if history:

        recent_history = history[
            -(MAX_HISTORY * 2):
        ]

        # Basic validation so malformed frontend history
        # does not break the request.
        for item in recent_history:

            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")

            if role not in (
                "user",
                "assistant"
            ):
                continue

            if not content:
                continue

            messages.append(
                {
                    "role": role,
                    "content": str(content)
                }
            )

    # ---------------------------------------------------------
    # Current user message
    # ---------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    return messages


# =============================================================================
# BOOKING INTENT
# =============================================================================

def is_booking_intent(message):
    """
    Detect whether the user wants to book an appointment.
    """

    message_lower = message.lower()

    return any(
        keyword in message_lower
        for keyword in BOOKING_KEYWORDS
    )


# =============================================================================
# HOME ROUTE
# =============================================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.route("/health")
def health():

    try:

        store = get_vector_store()

        return jsonify(
            {
                "status": "ok",
                "chunks_in_db": len(store),
                "chat_model": CHAT_MODEL,
                "embed_model": EMBED_MODEL
            }
        )

    except Exception as e:

        logger.error(
            f"Health check error: {e}",
            exc_info=True
        )

        return jsonify(
            {
                "status": "error",
                "message": str(e)
            }
        ), 200


# =============================================================================
# CHAT ROUTE
# =============================================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    request_start = time.perf_counter()

    try:

        # ---------------------------------------------------------
        # Read request
        # ---------------------------------------------------------

        data = request.get_json(
            silent=True
        ) or {}

        message = (
            data.get("message")
            or ""
        ).strip()

        history = (
            data.get("history")
            or []
        )

        if not message:

            return jsonify(
                {
                    "error":
                    "No message provided"
                }
            ), 400

        logger.info(
            f"Incoming message: {message}"
        )

        # ---------------------------------------------------------
        # BOOKING INTENT
        #
        # IMPORTANT:
        # Booking requests do NOT go through RAG or the LLM.
        # The frontend can immediately display the form.
        # ---------------------------------------------------------

        show_booking_form = is_booking_intent(
            message
        )

        if show_booking_form:

            logger.info(
                "Booking intent detected — "
                "returning booking response immediately."
            )

            total_time = (
                time.perf_counter()
                - request_start
            )

            logger.info(
                f"Booking response completed "
                f"in {total_time:.2f}s."
            )

            return jsonify(
                {
                    "reply": (
                        "Absolutely. I'll help you "
                        "get started with an appointment. "
                        "Please fill in the quick "
                        "appointment form."
                    ),
                    "sources": [],
                    "show_booking_form": True
                }
            )

        # ---------------------------------------------------------
        # VECTOR SEARCH
        # ---------------------------------------------------------

        results = retrieve_results(
            message
        )

        # ---------------------------------------------------------
        # BUILD RAG CONTEXT
        # ---------------------------------------------------------

        context = build_context(
            results
        )

        # ---------------------------------------------------------
        # BUILD SOURCES
        # ---------------------------------------------------------

        sources = build_sources(
            results
        )

        # ---------------------------------------------------------
        # BUILD CHAT MESSAGES
        # ---------------------------------------------------------

        messages = build_messages(
            history,
            context,
            message
        )

        # ---------------------------------------------------------
        # Log total chat request size
        # ---------------------------------------------------------

        total_message_chars = sum(
            len(
                str(
                    item.get(
                        "content",
                        ""
                    )
                )
            )
            for item in messages
        )

        logger.info(
            f"Chat request size: "
            f"{total_message_chars} characters."
        )

        logger.info(
            f"Chat message count: "
            f"{len(messages)}."
        )

        # ---------------------------------------------------------
        # CALL CHAT MODEL
        # ---------------------------------------------------------

        logger.info(
            "Sending request to chat model..."
        )

        chat_start = time.perf_counter()

        response = (
            get_openai()
            .chat
            .completions
            .create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.4,
                max_tokens=300,
                extra_body={"chat_template_kwargs": {"thinking": False}}
            )
        )

        chat_elapsed = (
            time.perf_counter()
            - chat_start
        )

        logger.info(
            f"Chat model responded in "
            f"{chat_elapsed:.2f}s."
        )

        # ---------------------------------------------------------
        # Extract response
        # ---------------------------------------------------------

        if not response.choices:

            raise RuntimeError(
                "Chat model returned no choices."
            )

        reply = (
            response
            .choices[0]
            .message
            .content
        )

        if not reply:

            raise RuntimeError(
                "Chat model returned an empty response."
            )

        # ---------------------------------------------------------
        # Complete request
        # ---------------------------------------------------------

        total_time = (
            time.perf_counter()
            - request_start
        )

        logger.info(
            f"Request completed in "
            f"{total_time:.2f}s."
        )

        # ---------------------------------------------------------
        # Return response
        # ---------------------------------------------------------

        return jsonify(
            {
                "reply": reply,
                "sources": sources,
                "show_booking_form": False
            }
        )

    except Exception as e:

        total_time = (
            time.perf_counter()
            - request_start
        )

        logger.error(
            f"Chat error after "
            f"{total_time:.2f}s: {e}",
            exc_info=True
        )

        return jsonify(
            {
                "reply": (
                    "I'm experiencing a "
                    "temporary technical issue. "
                    "Please call or WhatsApp "
                    "+254 112 268 709."
                ),
                "sources": [],
                "show_booking_form": False
            }
        ), 200


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        logger.error(
            "OPENAI_API_KEY not found "
            "in .env file."
        )

        input(
            "Press Enter to close..."
        )

        exit(1)

    if not os.path.exists(
        VECTOR_STORE
    ):

        logger.error(
            "vector_store.json not found. "
            "Run python ingest.py first."
        )

        input(
            "Press Enter to close..."
        )

        exit(1)

    logger.info(
        "=== URO-CARE RAG Chatbot ==="
    )

    logger.info(
        f"Chat model:   {CHAT_MODEL}"
    )

    logger.info(
        f"Embed model:  {EMBED_MODEL}"
    )

    logger.info(
        f"Vector store: "
        f"{os.path.abspath(VECTOR_STORE)}"
    )

    logger.info(
        "Open browser at: "
        "http://127.0.0.1:5000"
    )

    logger.info(
        "Press CTRL+C to stop"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )

