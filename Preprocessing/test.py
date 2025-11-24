#!/usr/bin/env python3
"""
pipeline_no_chunking.py

Runs metadata enrichment, language/transliteration tagging, BM25 build,
embeddings+index (FAISS/Annoy fallback) and smoke tests **without** chunking.

Assumes you already have a passages JSONL at:
  NCERT_passages_hybrid/passages.jsonl

Outputs go to NCERT_processed/ and NCERT_passages_hybrid/.

Usage examples:
  # Quick run using existing passages (embed a sample of 500 passages)
  python pipeline_no_chunking.py --embed_limit 500

  # Filter by grade (only use passages from Grade_12)
  python pipeline_no_chunking.py --grade Grade_12 --embed_limit 500

  # Use all passages (may be slow)
  python pipeline_no_chunking.py
"""
import argparse, json, csv, re, time, pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

# INPUT (your existing chunk file)
PASSAGES_JSONL = Path("NCERT_passages_hybrid/passages.jsonl")
PASSAGE_INDEX_CSV = Path("NCERT_passages_hybrid/passage_index.csv")

# OUTPUTS
PROCESSED = Path("NCERT_processed")
PROCESSED.mkdir(parents=True, exist_ok=True)
META_JSONL = PROCESSED / "passages_meta.jsonl"
META_CSV = PROCESSED / "passages_meta.csv"
META_LANG_JSONL = PROCESSED / "passages_meta_lang.jsonl"
META_LANG_CSV = PROCESSED / "passages_meta_lang.csv"
BM25_PICKLE = Path("NCERT_passages_hybrid/bm25.pkl")
EMB_NPY = PROCESSED / "passages_embs.npy"
EMB_META_JSON = PROCESSED / "passages_emb_meta.json"
FAISS_INDEX = PROCESSED / "passages.faiss"
ANNOY_INDEX = PROCESSED / "passages.ann"
SMOKE_DIR = PROCESSED / "smoke_test_results"
SMOKE_DIR.mkdir(parents=True, exist_ok=True)

# defaults
EMBED_BATCH = 16
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ---------------- Helpers ----------------
def safe_read_jsonl(path):
    recs = []
    if not path.exists():
        return recs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                recs.append(json.loads(line))
            except Exception:
                # skip bad lines
                continue
    return recs

# ---------------- Step 4: Metadata enrichment ----------------
def enrich_metadata(passages_path=PASSAGES_JSONL, out_jsonl=META_JSONL, out_csv=META_CSV, grade=None, subject=None):
    if not passages_path.exists():
        print("ERROR: passages JSONL not found at", passages_path)
        return 0
    written = 0
    rows = []
    with passages_path.open("r", encoding="utf-8") as inf, out_jsonl.open("w", encoding="utf-8") as outf:
        for line in inf:
            j = json.loads(line)
            if grade and j.get("grade") != grade: continue
            if subject and j.get("subject") != subject: continue
            text = j.get("text","")
            char_count = len(text)
            token_count = len(re.findall(r"\S+", text))
            # basic script profile
            counts = {"latin":0, "devanagari":0, "arabic":0, "other":0}
            for ch in text:
                o = ord(ch)
                if 0x0900 <= o <= 0x097F: counts["devanagari"] += 1
                elif (0x0600 <= o <= 0x06FF) or (0x0750 <= o <= 0x077F): counts["arabic"] += 1
                elif (65 <= o <= 90) or (97 <= o <=122) or (48 <= o <=57): counts["latin"] += 1
                else: counts["other"] += 1
            total = sum(counts.values()) or 1
            pct = {k: counts[k]/total for k in counts}
            if pct["devanagari"] > 0.6:
                primary = "hi"
            elif pct["arabic"] > 0.6:
                primary = "ur"
            elif pct["latin"] > 0.6:
                primary = "en"
            else:
                primary = "mixed"
            meta = {
                "id": j.get("id"),
                "grade": j.get("grade"),
                "subject": j.get("subject"),
                "chapter": j.get("chapter"),
                "source": j.get("source"),
                "char_count": char_count,
                "token_count": token_count,
                "script_counts": counts,
                "script_pct": pct,
                "primary_language": primary,
                "difficulty": "unknown",
                "source_page_range": None,
                "text": text
            }
            outf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            rows.append({"id":meta["id"], "grade":meta["grade"], "subject":meta["subject"], "chapter":meta["chapter"], "token_count":meta["token_count"], "primary_language":meta["primary_language"]})
            written += 1
    # CSV
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    print("Metadata enriched for", written, "passages")
    return written

# ---------------- Step 5: Language + transliteration tagging ----------------
def lang_translit(in_jsonl=META_JSONL, out_jsonl=META_LANG_JSONL, out_csv=META_LANG_CSV):
    if not Path(in_jsonl).exists():
        print("ERROR: metadata JSONL not found at", in_jsonl)
        return 0
    rows=[]
    with Path(in_jsonl).open("r", encoding="utf-8") as inf, Path(out_jsonl).open("w", encoding="utf-8") as outf:
        for line in inf:
            j = json.loads(line)
            pct = j.get("script_pct", {})
            subject = j.get("subject","")
            primary = j.get("primary_language","unknown")
            scripts_over = [k for k,v in pct.items() if v > 0.15]
            mixed = len(scripts_over) >= 2
            translit = "none"
            if primary == "en" and subject and subject.lower() in ["hindi","sanskrit","urdu"]:
                translit = "romanized"
            if primary == "mixed":
                translit = "mixed"
            j["transliteration_tag"] = translit
            j["mixed_language"] = mixed
            outf.write(json.dumps(j, ensure_ascii=False) + "\n")
            rows.append({"id":j["id"], "grade":j["grade"], "subject":j["subject"], "chapter":j["chapter"], "primary_language":j["primary_language"], "transliteration_tag":j["transliteration_tag"], "mixed_language":j["mixed_language"]})
    if rows:
        with Path(out_csv).open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    print("Language tagging done for", len(rows), "passages")
    return len(rows)

# ---------------- Step 6: BM25 ----------------
def build_bm25(passages_jsonl=PASSAGES_JSONL, out_pickle=BM25_PICKLE, grade=None, subject=None):
    print("Building BM25")
    try:
        from rank_bm25 import BM25Okapi
    except Exception as e:
        print("Error: rank_bm25 not installed:", e)
        return False
    records=[]; tokenized=[]
    with Path(passages_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            j=json.loads(line)
            if grade and j.get("grade") != grade: continue
            if subject and j.get("subject") != subject: continue
            records.append(j)
            tokenized.append([t.lower() for t in re.findall(r'\w+', j.get("text",""))])
    if not records:
        print("No records found for BM25")
        return False
    bm25 = BM25Okapi(tokenized)
    with out_pickle.open("wb") as pf:
        pickle.dump({"bm25":bm25, "records":records}, pf)
    print("BM25 saved to", out_pickle, "with", len(records), "records")
    return True

# ---------------- Step 7: Embeddings + FAISS/Annoy ----------------
def embed_and_index(meta_lang_jsonl=None, embed_limit=None, batch=16, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    """
    Embeds passages from meta-lang JSONL (default: NCERT_processed/passages_meta_lang.jsonl)
    and writes embeddings + FAISS/Annoy index.
    """
    # resolve default if caller passed None
    if meta_lang_jsonl is None:
        meta_lang_jsonl = Path("NCERT_processed/passages_meta_lang.jsonl")
    else:
        meta_lang_jsonl = Path(meta_lang_jsonl)

    if not meta_lang_jsonl.exists():
        print(f"Error: meta-lang file not found at: {meta_lang_jsonl}")
        print("Make sure you ran the language tagging step and that the path is correct.")
        return False

    metas = []
    texts = []
    with meta_lang_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if embed_limit and len(texts) >= embed_limit:
                break
            if not line.strip():
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            metas.append({k: j.get(k) for k in ("id","grade","subject","chapter","source","token_count")})
            texts.append(j.get("text",""))

    if not texts:
        print("No passages to embed (after filtering).")
        return False

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
    except Exception as e:
        print("Error loading sentence-transformers:", e)
        return False

    import numpy as np
    from tqdm import tqdm
    arrs = []
    for i in tqdm(range(0, len(texts), batch), desc="Embedding batches"):
        batch_texts = texts[i:i+batch]
        emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        arrs.append(emb.astype("float32"))
    embs = np.vstack(arrs)

    # save embeddings + meta
    Path("NCERT_processed").mkdir(parents=True, exist_ok=True)
    np.save("NCERT_processed/passages_embs.npy", embs)
    Path("NCERT_processed/passages_emb_meta.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved embeddings and meta (NCERT_processed/)")

    # try faiss, else annoy
    try:
        import faiss
        d = embs.shape[1]
        idx = faiss.IndexFlatL2(d)
        idx.add(embs)
        faiss.write_index(idx, "NCERT_processed/passages.faiss")
        print("Saved FAISS index: NCERT_processed/passages.faiss")
        return "faiss"
    except Exception as e:
        print("FAISS failed (will attempt Annoy):", e)
        try:
            from annoy import AnnoyIndex
            d = embs.shape[1]
            t = AnnoyIndex(d, metric='euclidean')
            for ii, v in enumerate(embs):
                t.add_item(ii, v.tolist())
            t.build(10)
            t.save("NCERT_processed/passages.ann")
            print("Saved Annoy index: NCERT_processed/passages.ann")
            return "annoy"
        except Exception as e2:
            print("Annoy failed too:", e2)
            return False

# ---------------- Step 8: Smoke tests ----------------
def smoke_test(topk=5, queries=None, embed_model=EMBED_MODEL):
    print("Running smoke tests")
    if BM25_PICKLE.exists():
        bm = pickle.load(open(BM25_PICKLE,"rb"))
    else:
        print("BM25 pickle not found:", BM25_PICKLE)
        bm = None

    # load FAISS/Annoy if available
    faiss_idx=None; annoy_idx=None; embs=None; meta=None
    if FAISS_INDEX.exists():
        try:
            import faiss
            faiss_idx = faiss.read_index(str(FAISS_INDEX))
            meta = json.loads(EMB_META_JSON.read_text(encoding="utf-8"))
            embs = np.load(EMB_NPY)
        except Exception as e:
            print("Failed to load FAISS:", e)
            faiss_idx = None
    elif ANNOY_INDEX.exists():
        try:
            from annoy import AnnoyIndex
            meta = json.loads(EMB_META_JSON.read_text(encoding="utf-8"))
            embs = np.load(EMB_NPY)
            annoy_idx = AnnoyIndex(embs.shape[1], metric='euclidean')
            annoy_idx.load(str(ANNOY_INDEX))
        except Exception as e:
            print("Failed to load Annoy:", e)
            annoy_idx = None

    if not queries:
        queries = ["What is photosynthesis?", "Explain Pythagoras theorem.", "पर्यावरण संरक्षण के उपाय क्या हैं", "Describe the digestive system"]

    results=[]
    # load embedding model for vector queries if needed
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embed_model)
    except Exception:
        model = None

    for q in queries:
        entry = {"query": q, "bm25": [], "vector": []}
        if bm:
            toks = [t.lower() for t in re.findall(r'\w+', q)]
            scores = bm["bm25"].get_scores(toks)
            topn = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:topk]
            for rank,(idx,sc) in enumerate(topn, start=1):
                rec = bm["records"][idx]
                entry["bm25"].append({"rank": rank, "id": rec.get("id"), "path": rec.get("source"), "score": float(sc)})
        if faiss_idx is not None and model is not None:
            try:
                qemb = model.encode([q], convert_to_numpy=True).astype("float32")
                D,I = faiss_idx.search(qemb, topk)
                for rank,(dist, idxv) in enumerate(zip(D[0], I[0]), start=1):
                    if idxv < 0: continue
                    rec = meta[idxv]
                    entry["vector"].append({"rank": rank, "id": rec.get("id"), "path": rec.get("source"), "dist": float(dist)})
            except Exception as e:
                print("FAISS query failed:", e)
        elif annoy_idx is not None and model is not None:
            try:
                qemb = model.encode([q], convert_to_numpy=True)[0].tolist()
                ids, dists = annoy_idx.get_nns_by_vector(qemb, topk, include_distances=True)
                for rank,(iidx, dist) in enumerate(zip(ids, dists), start=1):
                    rec = meta[iidx]
                    entry["vector"].append({"rank": rank, "id": rec.get("id"), "path": rec.get("source"), "dist": float(dist)})
            except Exception as e:
                print("Annoy query failed:", e)
        results.append(entry)

    outp = SMOKE_DIR / "smoke_results.json"
    outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Smoke test results written to", outp)
    return results

# ---------------- Orchestrator ----------------
def run_all(grade=None, subject=None, embed_limit=None):
    start=time.time()
    cnt = enrich_metadata(grade=grade, subject=subject)
    lang_cnt = lang_translit()
    bm25_ok = build_bm25(grade=grade, subject=subject)
    idx_mode = embed_and_index(embed_limit)
    smoke = smoke_test()
    elapsed=time.time()-start
    print("Pipeline finished in %.1f s. embed mode: %s" % (elapsed, str(idx_mode)))
    return {"meta":cnt, "lang":lang_cnt, "bm25":bm25_ok, "index_mode":idx_mode}

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grade", type=str, help="filter passages by grade")
    p.add_argument("--subject", type=str, help="filter passages by subject")
    p.add_argument("--embed_limit", type=int, help="limit number of passages to embed (sample)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_all(grade=args.grade, subject=args.subject, embed_limit=args.embed_limit)
