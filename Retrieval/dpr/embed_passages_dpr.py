#!/usr/bin/env python3
"""
embed_passages_dpr.py

Task B — DPR (Dense Passage Retrieval) dual-encoder setup.

What this does:
  - Loads passages from:
        Preprocessing/NCERT_paragraphs/passages.jsonl
    (each line is a JSON object with at least: id, text, grade, subject, chapter)
  - Encodes all passages using a SentenceTransformer model (dual-encoder style).
  - Saves:
        Retrieval/dpr/passages_dpr_embs.npy      (float32 [N, D])
        Retrieval/dpr/passages_dpr_meta.json     (list of {id, grade, subject, chapter, idx})

These embeddings will be used in:
  - Task C: FAISS index build & tuning
  - Task D/F/H: Retrieval, hybrid search, evaluation.

Usage (from project root):
  cd D:\eduniti-majorProject
  python Retrieval\\dpr\\embed_passages_dpr.py
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

# -------- CONFIG --------

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # points to eduniti-majorProject

# adjust this depending on whether you're using NCERT_paragraphs or NCERT_passages_hybrid
PASSAGES_JSONL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "passages.jsonl"
# If you're using the hybrid chunks instead, comment the above line and uncomment this:
# PASSAGES_JSONL = PROJECT_ROOT / "Preprocessing" / "NCERT_passages_hybrid" / "passages.jsonl"

OUT_DIR = PROJECT_ROOT / "Retrieval" / "dpr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_OUT = OUT_DIR / "passages_dpr_embs.npy"
META_OUT = OUT_DIR / "passages_dpr_meta.json"

# Pretrained dual-encoder style model
# Options:
#   - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (good multilingual)
#   - "sentence-transformers/multi-qa-mpnet-base-dot-v1" (QA-oriented, English-heavy)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

BATCH_SIZE = 32
MAX_PASSAGES = None   # set e.g. 5000 to test on a subset


# -------- LOAD PASSAGES --------

def load_passages(path: Path, limit: int | None = None):
    """
    Load passages from JSONL.
    Each line expected to be:
      {
        "id": "...",
        "text": "...",
        "grade": "...",
        "subject": "...",
        "chapter": "...",
        ...
      }
    Returns:
      texts: list[str]
      metas: list[dict] (id, grade, subject, chapter, idx)
    """
    if not path.exists():
        raise FileNotFoundError(f"Passages JSONL not found at: {path.resolve()}")

    texts = []
    metas = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            pid = obj.get("id")
            txt = obj.get("text", "")
            grade = obj.get("grade", "")
            subject = obj.get("subject", "")
            chapter = obj.get("chapter", "")

            if not pid or not txt.strip():
                continue

            metas.append({
                "id": pid,
                "grade": grade,
                "subject": subject,
                "chapter": chapter,
                "idx": len(texts)   # index in embedding array
            })
            texts.append(txt)

            if limit is not None and len(texts) >= limit:
                break

    return texts, metas


# -------- ENCODING --------

def encode_passages(texts):
    """
    Encode all passages using SentenceTransformer.
    Returns numpy array [N, D] float32.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading dual-encoder model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding passages"):
        batch = texts[i:i+BATCH_SIZE]
        embs = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE
        ).astype("float32")
        all_embs.append(embs)

    if not all_embs:
        return np.zeros((0, 0), dtype="float32")

    embs = np.vstack(all_embs)
    return embs


# -------- MAIN --------

def main():
    print("➡️ Task B — DPR dual-encoder setup: embedding passages...")

    texts, metas = load_passages(PASSAGES_JSONL, limit=MAX_PASSAGES)
    print(f"Loaded {len(texts)} passages for embedding.")

    if not texts:
        print("No passages loaded. Check PASSAGES_JSONL path or preprocessing output.")
        return

    embs = encode_passages(texts)
    print(f"Embeddings shape: {embs.shape}")

    # Save embeddings + meta
    np.save(EMB_OUT, embs)
    META_OUT.write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Saved embeddings to: {EMB_OUT}")
    print(f"✅ Saved meta to:       {META_OUT}")
    print("These will be used in Task C (FAISS index build).")


if __name__ == "__main__":
    main()
