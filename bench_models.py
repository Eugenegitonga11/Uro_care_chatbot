"""
bench_models.py — pick a chat model for the URO-CARE bot

Runs each candidate model against a realistic payload: your actual
knowledge-base chunks as context, plus a real patient question. Reports
latency, whether the model wastes tokens on hidden reasoning, and the
reply itself so you can judge tone.

Usage:

    # 1. See what's available, grouped sensibly
    python bench_models.py --list

    # 2. Benchmark the ones that look promising
    python bench_models.py meta/llama-3.3-70b-instruct mistralai/mistral-nemo-12b-instruct

    # 3. Or just try everything that looks like a chat model (slow)
    python bench_models.py --auto
"""

import json
import os
import sys
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
VECTOR_STORE = "./vector_store.json"
TIMEOUT = 60.0

# A real question a patient would ask, answerable from the knowledge base.
TEST_QUESTION = "Do I need a referral to see a urologist, and how much does a first consultation cost?"

# Words that mark a model as not a general chat model.
SKIP = (
    "embed", "rerank", "retriev", "guard", "ocr", "asr", "tts",
    "riva", "parakeet", "canary", "vila", "vlm", "vision",
    "diffusion", "flux", "sana", "stable-", "clip", "dino",
    "molmo", "protein", "fold", "genmol", "esm", "audio",
)

SYSTEM_TEMPLATE = """
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

--- KNOWLEDGE BASE CONTEXT ---

{context}

--- END KNOWLEDGE BASE CONTEXT ---
"""


def client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("FAIL: OPENAI_API_KEY not found in .env")
        sys.exit(1)
    return OpenAI(
        api_key=api_key,
        base_url=NVIDIA_BASE_URL,
        timeout=TIMEOUT,
        max_retries=0,
    )


def build_context():
    """Use real chunks so the prompt size matches production."""
    if not os.path.exists(VECTOR_STORE):
        return "No knowledge base found — running with a short prompt."

    with open(VECTOR_STORE, "r", encoding="utf-8") as f:
        store = json.load(f)

    parts = []
    for item in store[:5]:
        parts.append(
            f"[Section: {item.get('section', 'General')}]\n"
            f"{item.get('text', '')}"
        )
    return "\n\n---\n\n".join(parts)


def list_models(c):
    ids = sorted(m.id for m in c.models.list().data)

    chat, other = [], []
    for m in ids:
        (other if any(s in m.lower() for s in SKIP) else chat).append(m)

    print(f"\n{len(chat)} likely chat models:\n")
    for m in chat:
        print(f"  {m}")

    print(f"\n{len(other)} skipped (embedding / vision / audio / etc):\n")
    for m in other:
        print(f"  {m}")

    return chat


def bench(c, model, messages):
    start = time.perf_counter()
    try:
        resp = c.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            max_tokens=300,
        )
    except Exception as e:
        return {
            "model": model,
            "ok": False,
            "secs": time.perf_counter() - start,
            "note": f"{type(e).__name__}: {str(e)[:90]}",
            "reply": "",
        }

    secs = time.perf_counter() - start
    choice = resp.choices[0] if resp.choices else None
    content = (choice.message.content or "") if choice else ""
    reasoning = getattr(choice.message, "reasoning_content", None) if choice else None

    note = ""
    if reasoning:
        note = f"REASONING ({len(reasoning)} chars) — avoid"
    elif not content:
        note = "EMPTY reply — avoid"
    elif choice and choice.finish_reason == "length":
        note = "truncated at 300 tokens"

    return {
        "model": model,
        "ok": bool(content) and not reasoning,
        "secs": secs,
        "note": note,
        "reply": content.strip(),
    }


def main():
    args = [a for a in sys.argv[1:]]
    c = client()

    if "--list" in args:
        list_models(c)
        return

    if "--auto" in args:
        candidates = list_models(c)
        print(f"\nBenchmarking all {len(candidates)} — this will take a while.\n")
    else:
        candidates = args

    if not candidates:
        print(__doc__)
        return

    context = build_context()
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)},
        {"role": "user", "content": TEST_QUESTION},
    ]

    size = sum(len(m["content"]) for m in messages)
    print(f"Payload: {size} characters (production is ~9,200)")
    print(f"Question: {TEST_QUESTION}\n")

    results = []
    for model in candidates:
        print(f"Testing {model} ... ", end="", flush=True)
        r = bench(c, model, messages)
        results.append(r)
        flag = "OK " if r["ok"] else "-- "
        print(f"{flag}{r['secs']:.2f}s  {r['note']}")

    print("\n" + "=" * 70)
    print("RANKED (usable models, fastest first)")
    print("=" * 70)

    usable = sorted([r for r in results if r["ok"]], key=lambda r: r["secs"])

    if not usable:
        print("\nNothing usable. Try more candidates from --list.")
        return

    for r in usable:
        print(f"\n{r['secs']:6.2f}s  {r['model']}")
        if r["note"]:
            print(f"         note: {r['note']}")
        print(f"         {r['reply'][:400]}")

    print("\n" + "=" * 70)
    print(f"Fastest usable: {usable[0]['model']}")
    print("Read the replies above before deciding — speed is not the only")
    print("thing that matters. Check it stays factual, refuses to diagnose,")
    print("and sounds warm rather than robotic.")
    print("=" * 70)


if __name__ == "__main__":
    main()