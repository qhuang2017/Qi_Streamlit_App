import os, pickle, math
from pypdf import PdfReader
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
import tiktoken
import faiss

# ---------- SETTINGS ----------
DATA_DIR = Path("course_materials")     # put your PDFs/TXT here
INDEX_PATH = Path("index.faiss")
CHUNKS_PATH = Path("chunks.pkl")
EMBED_MODEL = "text-embedding-3-small"  # inexpensive, good enough
CHUNK_SIZE = 900                         # ~900 characters per chunk
CHUNK_OVERLAP = 150                      # overlap text a bit for continuity
# --------------------------------

def load_text_from_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    else:
        return path.read_text(encoding="utf-8", errors="ignore")

def chunk_text(text: str, source: str) -> List[Dict]:
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+CHUNK_SIZE]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source})
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def main():
    assert DATA_DIR.exists(), f"Folder {DATA_DIR} not found."
    files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() in [".pdf", ".txt", ".md"]]
    assert files, f"No .pdf/.txt/.md files found in {DATA_DIR}"

    # Gather & chunk
    all_chunks = []
    for f in files:
        print(f"Reading {f.name} ...")
        raw = load_text_from_file(f)
        all_chunks.extend(chunk_text(raw, f.name))

    print(f"Total chunks: {len(all_chunks)}")

    # Create embeddings
    client = OpenAI()  # reads OPENAI_API_KEY from environment or Streamlit secrets at runtime
    texts = [c["text"] for c in all_chunks]

    # Batch to avoid large payloads
    batch_size = 100
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
        print(f"Embedded {min(start+batch_size, len(texts))}/{len(texts)}")

    # Build FAISS index
    import numpy as np
    vecs = np.array(vectors).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])  # inner product
    # Normalize for cosine similarity equivalence
    faiss.normalize_L2(vecs)
    index.add(vecs)

    # Save index and chunks
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("Index built! Files saved:", INDEX_PATH, CHUNKS_PATH)

if __name__ == "__main__":
    main()