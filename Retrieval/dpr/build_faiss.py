"""
Task C - FAISS index build & tuning for DPR embeddings.

- Loads passage embeddings from: passages_dpr_embs.npy
- Optionally L2-normalizes them (for dot-product / cosine search)
- Builds:
    1) Flat index        (IndexFlatIP)
    2) IVF-Flat index    (IndexIVFFlat + IndexFlatIP quantizer)
    3) HNSW-Flat index   (IndexHNSWFlat)
- Saves indexes to disk for later use in retrieval.
"""

import os
from pathlib import Path

import faiss
import numpy as np
import json

# ----------------- PATHS -----------------

BASE_DIR = Path(r"D:/eduniti-majorProject/Retrieval/dpr")  # adjust if needed

EMB_PATH = BASE_DIR / "passages_dpr_embs.npy"
META_PATH = BASE_DIR / "passages_dpr_meta.json"

INDEX_DIR = BASE_DIR / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FLAT_INDEX_PATH = INDEX_DIR / "dpr_faiss_flat.index"
IVF_INDEX_PATH = INDEX_DIR / "dpr_faiss_ivf.index"
HNSW_INDEX_PATH = INDEX_DIR / "dpr_faiss_hnsw.index"


# ----------------- LOAD EMBEDDINGS -----------------

def load_embeddings(path: Path) -> np.ndarray:
    print(f"🔹 Loading embeddings from: {path}")
    embs = np.load(path)
    print(f"   Shape: {embs.shape} (num_passages, dim)")
    return embs.astype("float32")  # faiss prefers float32


def maybe_load_meta(path: Path):
    if not path.exists():
        print(f"⚠️ Meta file not found at {path}, continuing without it.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"🔹 Loaded meta for {len(meta)} passages.")
    return meta


# ----------------- NORMALIZATION -----------------

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length.
    DPR-style / sentence-transformers retrieval usually uses dot product
    or cosine similarity; with IndexFlatIP we typically normalize.
    """
    faiss.normalize_L2(x)
    return x


# ----------------- FLAT INDEX (Exact Search) -----------------

def build_flat_index(embs: np.ndarray, normalize: bool = True) -> faiss.Index:
    dim = embs.shape[1]
    xb = embs.copy()
    if normalize:
        print("✅ L2-normalizing embeddings for Flat IP index...")
        l2_normalize(xb)

    print(f"🔧 Building Flat index (IndexFlatIP) with dim={dim} ...")
    index = faiss.IndexFlatIP(dim)
    index.add(xb)
    print(f"   → Added {index.ntotal} vectors to Flat index.")
    return index


# ----------------- IVF-Flat INDEX (Approximate) -----------------

def build_ivf_index(embs: np.ndarray,
                    nlist: int = 256,
                    nprobe: int = 16,
                    normalize: bool = True) -> faiss.IndexIVFFlat:
    """
    IVF-Flat: coarse quantizer + inverted lists.
    - nlist: number of clusters (try 256, 512, 1024; tune later)
    - nprobe: number of clusters to probe at query time (tradeoff speed/recall)
    """
    dim = embs.shape[1]
    xb = embs.copy()
    if normalize:
        print("✅ L2-normalizing embeddings for IVF IP index...")
        l2_normalize(xb)

    print(f"🔧 Building IVF-Flat index with dim={dim}, nlist={nlist}, nprobe={nprobe} ...")
    quantizer = faiss.IndexFlatIP(dim)  # inner product quantizer
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # IVF requires training on samples
    print("   → Training IVF index...")
    index_ivf.train(xb)
    print("   → Adding vectors to IVF index...")
    index_ivf.add(xb)

    # Set nprobe (how many clusters to search)
    index_ivf.nprobe = nprobe
    print(f"   → IVF index built with {index_ivf.ntotal} vectors.")
    return index_ivf


# ----------------- HNSW-Flat INDEX (Graph-based ANN) -----------------

def build_hnsw_index(embs: np.ndarray,
                     M: int = 32,
                     ef_construction: int = 200,
                     ef_search: int = 50,
                     normalize: bool = True) -> faiss.IndexHNSWFlat:
    """
    HNSW-Flat:
    - M: number of connections per node (higher → better recall, slower)
    - ef_construction: controls index build complexity/accuracy
    - ef_search: controls search-time accuracy; can tune at query time too
    """
    dim = embs.shape[1]
    xb = embs.copy()
    if normalize:
        print("✅ L2-normalizing embeddings for HNSW IP index...")
        l2_normalize(xb)

    print(f"🔧 Building HNSW-Flat index with dim={dim}, M={M}, efC={ef_construction}, efS={ef_search} ...")
    index_hnsw = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efConstruction = ef_construction
    index_hnsw.hnsw.efSearch = ef_search

    print("   → Adding vectors to HNSW index...")
    index_hnsw.add(xb)
    print(f"   → HNSW index built with {index_hnsw.ntotal} vectors.")
    return index_hnsw


# ----------------- SAVE / LOAD HELPERS -----------------

def save_index(index: faiss.Index, path: Path):
    print(f"💾 Saving index → {path}")
    faiss.write_index(index, str(path))


def main():
    # 1. Load embeddings (and optionally metadata)
    embs = load_embeddings(EMB_PATH)
    _ = maybe_load_meta(META_PATH)  # not used here but good sanity check

    # 2. Build Flat index (exact search, baseline)
    flat_index = build_flat_index(embs, normalize=True)
    save_index(flat_index, FLAT_INDEX_PATH)

    # 3. Build IVF index (approximate, tunable)
    # nlist ~ sqrt(N) is a common heuristic; for ~27k vectors, ~256 is reasonable
    ivf_index = build_ivf_index(embs, nlist=256, nprobe=16, normalize=True)
    save_index(ivf_index, IVF_INDEX_PATH)

    # 4. Build HNSW index (graph-based ANN)
    hnsw_index = build_hnsw_index(embs, M=32, ef_construction=200, ef_search=50, normalize=True)
    save_index(hnsw_index, HNSW_INDEX_PATH)

    print("\n✅ Task C complete: all FAISS indexes built and saved.")
    print(f"   Flat index : {FLAT_INDEX_PATH}")
    print(f"   IVF index  : {IVF_INDEX_PATH}")
    print(f"   HNSW index : {HNSW_INDEX_PATH}")


if __name__ == "__main__":
    main()
