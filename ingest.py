"""
ingest.py  —  URO-CARE RAG Knowledge Base Ingestion
Uses a simple JSON vector store instead of ChromaDB (works on all Windows)
"""

import os
import re
import sys
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
KB_FILE         = "urocare_rag_knowledge_base.md"
VECTOR_STORE    = "./vector_store.json"
EMBED_MODEL     = "nvidia/nv-embedqa-e5-v5"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
CHUNK_SIZE      = 400
CHUNK_OVERLAP   = 60


def load_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_sections(text):
    pattern  = re.compile(r"^## (.+)$", re.MULTILINE)
    matches  = list(pattern.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        title   = match.group(1).strip()
        start   = match.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append({"title": title, "content": content})
    return sections


def chunk_text(text, max_words=CHUNK_SIZE, overlap_words=CHUNK_OVERLAP):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += max_words - overlap_words
    return chunks


def build_chunks(sections):
    all_chunks = []
    for sec in sections:
        full_text  = f"{sec['title']}\n\n{sec['content']}"
        sub_chunks = chunk_text(full_text)
        for j, chunk in enumerate(sub_chunks):
            safe_title = re.sub(r'[^a-z0-9]', '_', sec['title'].lower())[:40]
            all_chunks.append({
                "id":       f"{safe_title}_{j}",
                "text":     chunk,
                "section":  sec["title"],
                "chunk":    j,
            })
    return all_chunks


def get_embeddings(client, texts):
    embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        resp  = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "END"}
        )
        embeddings.extend([item.embedding for item in resp.data])
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...")
    return embeddings


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env file.")
        sys.exit(1)

    print("=== URO-CARE Knowledge Base Ingestion ===\n")

    # 1. Load & chunk
    print(f"Loading: {KB_FILE}")
    raw      = load_markdown(KB_FILE)
    raw      = "\n".join(l for l in raw.splitlines() if not l.startswith("#!"))
    sections = split_into_sections(raw)
    print(f"Sections: {len(sections)}")

    chunks = build_chunks(sections)
    print(f"Chunks:   {len(chunks)}")

    # 2. Embed
    print(f"\nGenerating embeddings via NVIDIA ({EMBED_MODEL})...")
    client     = OpenAI(api_key=api_key, base_url=NVIDIA_BASE_URL)
    texts      = [c["text"] for c in chunks]
    embeddings = get_embeddings(client, texts)
    print(f"Done. Dimensions: {len(embeddings[0])}")

    # 3. Save as JSON vector store
    print(f"\nSaving vector store to: {VECTOR_STORE}")
    store = []
    for chunk, emb in zip(chunks, embeddings):
        store.append({
            "id":        chunk["id"],
            "text":      chunk["text"],
            "section":   chunk["section"],
            "chunk":     chunk["chunk"],
            "embedding": emb,
        })

    with open(VECTOR_STORE, "w", encoding="utf-8") as f:
        json.dump(store, f)

    # Verify
    size_mb = os.path.getsize(VECTOR_STORE) / (1024 * 1024)
    print(f"Saved {len(store)} chunks ({size_mb:.1f} MB)")
    print(f"\n SUCCESS! Vector store ready.")
    print(f"Next step: python app.py")


if __name__ == "__main__":
    main()