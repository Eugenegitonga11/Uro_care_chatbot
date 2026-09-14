"""
check_models.py — URO-CARE chatbot diagnostics

Talks to the NVIDIA API directly, with Flask completely out of the way.
This tells you whether the problem is your models, your key, or your app.

Run from the project folder:
    python check_models.py
"""

import os
import sys
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Must match app.py exactly.
EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
CHAT_MODEL = "deepseek-ai/deepseek-v4-flash-0731"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

TIMEOUT = 60.0


def line():
    print("-" * 62)


def main():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("FAIL: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    print(f"Key loaded: {api_key[:10]}...{api_key[-4:]}")

    client = OpenAI(
        api_key=api_key,
        base_url=NVIDIA_BASE_URL,
        timeout=TIMEOUT,
        max_retries=0,
    )

    # ------------------------------------------------------------------
    # 1. Can we reach the API at all, and do our two models exist?
    # ------------------------------------------------------------------

    line()
    print("1. Listing available models")
    line()

    available = []

    try:
        start = time.perf_counter()
        models = client.models.list()
        available = sorted(m.id for m in models.data)
        print(f"OK — {len(available)} models visible "
              f"({time.perf_counter() - start:.2f}s)")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        print("\nIf this failed, the key or the network is the problem — "
              "stop here.")
        sys.exit(1)

    for label, model in (("EMBED", EMBED_MODEL), ("CHAT", CHAT_MODEL)):
        mark = "FOUND" if model in available else "NOT IN LIST"
        print(f"  {label:6} {model}  ->  {mark}")

    print("\nEmbedding-ish models on your account:")
    for m in available:
        if "embed" in m.lower():
            print(f"  {m}")

    # ------------------------------------------------------------------
    # 2. Embedding call
    # ------------------------------------------------------------------

    line()
    print("2. Embedding call")
    line()

    try:
        start = time.perf_counter()
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=["What are your opening hours?"],
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"},
        )
        dims = len(resp.data[0].embedding)
        print(f"OK — {dims} dimensions in "
              f"{time.perf_counter() - start:.2f}s")
        print("     (vector_store.json holds 2048-dim vectors — "
              "these must match)")
    except Exception as e:
        print(f"FAIL after {time.perf_counter() - start:.2f}s: "
              f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # 3. Chat call — the piece most likely to be hanging
    # ------------------------------------------------------------------

    line()
    print("3. Chat call")
    line()

    try:
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "Say hello in one line."}],
            temperature=0.4,
            max_tokens=300,
        )
        elapsed = time.perf_counter() - start

        choice = resp.choices[0] if resp.choices else None
        content = choice.message.content if choice else None
        reasoning = getattr(choice.message, "reasoning_content", None) \
            if choice else None

        print(f"Responded in {elapsed:.2f}s")
        print(f"  finish_reason : {choice.finish_reason if choice else 'n/a'}")
        print(f"  content       : {content!r}")

        if reasoning:
            print(f"  reasoning_content present "
                  f"({len(reasoning)} chars)")
            print("  ^ This is a reasoning model. If content is empty while "
                  "reasoning_content\n    is full, max_tokens=300 is being "
                  "spent on thinking. Raise max_tokens\n    or disable "
                  "thinking.")

        if not content:
            print("\n  WARNING: empty content — app.py raises on this and "
                  "shows the\n  'temporary technical issue' message.")

    except Exception as e:
        print(f"FAIL after {time.perf_counter() - start:.2f}s: "
              f"{type(e).__name__}: {e}")

    line()
    print("Done. If all three steps pass quickly, the API is fine and the "
          "hang is\nin how the app is being served (see gunicorn timeout / "
          "console freeze).")
    line()


if __name__ == "__main__":
    main()