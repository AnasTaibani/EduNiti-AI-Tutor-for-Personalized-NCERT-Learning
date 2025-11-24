#!/usr/bin/env python3
"""
train_dpr_ncert.py  (FAST CPU-FRIENDLY VERSION)

Fine-tune DPR-style dual encoder on NCERT gold pairs using SentenceTransformers.

Inputs:
  - Retrieval/dpr/train_pairs.jsonl
      one JSON per line: { "query": ..., "passage_id": ..., "passage_text": ... }

Output:
  - Retrieval/dpr/models/ncert-dpr-v1/   (SentenceTransformer model directory)
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PAIRS = PROJECT_ROOT / "Retrieval" / "dpr" / "train_pairs.jsonl"
OUT_MODEL_DIR = PROJECT_ROOT / "Retrieval" / "dpr" / "models" / "ncert-dpr-v1"

# ✅ Smaller, faster multilingual model (better for CPU)
# BASE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # OLD (slow)
BASE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# or even smaller if needed:
# BASE_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v1"

# ✅ Training hyperparameters tuned for CPU
BATCH_SIZE = 4          # smaller batch = less RAM, faster step on CPU
NUM_EPOCHS = 1          # one pass over 247 pairs is enough for a light finetune
LR = 2e-5
WARMUP_RATIO = 0.1      # 10% warmup


def load_train_pairs(path: Path):
    """Load training pairs from JSONL into SentenceTransformers InputExample list."""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("query") or "").strip()
            p = (obj.get("passage_text") or "").strip()
            if not q or not p:
                continue
            examples.append(InputExample(texts=[q, p], label=1.0))
    return examples


def main():
    if not TRAIN_PAIRS.exists():
        print("ERROR: train_pairs.jsonl not found at", TRAIN_PAIRS)
        return

    print("Loading training pairs from:", TRAIN_PAIRS)
    train_examples = load_train_pairs(TRAIN_PAIRS)
    print("Total training pairs:", len(train_examples))
    if len(train_examples) < 10:
        print("⚠️ Very few training pairs; training might be unstable. Try adding more gold queries.")
    
    print("Loading base model:", BASE_MODEL_NAME)
    model = SentenceTransformer(BASE_MODEL_NAME)

    # ✅ Shorten sequence length for speed (most NCERT Q&A are short)
    model.max_seq_length = 128

    # ✅ DataLoader with small batch, no pin_memory (CPU)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE,
    )
    train_loss = losses.CosineSimilarityLoss(model)

    # ❌ Skip heavy IR evaluator for now (too slow on CPU, not essential)
    # If you want it back later, we can re-add with a lighter schedule.

    warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * WARMUP_RATIO)
    print(f"Warmup steps: {warmup_steps}")

    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting training (CPU-optimized)...")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=str(OUT_MODEL_DIR),
        optimizer_params={'lr': LR},
        show_progress_bar=True,
        use_amp=False,   # ✅ disable mixed precision on CPU
    )

    print("✅ Training complete.")
    print("✅ Saved fine-tuned model to:", OUT_MODEL_DIR)


if __name__ == "__main__":
    main()
