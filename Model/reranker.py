#!/usr/bin/env python3
"""
reranker.py

BGE Cross-Encoder reranker wrapper.

- Takes BM25 hits (list of dicts with "text")
- Returns same hits sorted by cross-encoder score, with added "rerank_score".
"""

from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BGEReranker:
    """
    Thin wrapper around BAAI/bge-reranker-base for reranking BM25 hits.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
    ) -> None:
        # Decide device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        print(f"[Reranker] Loading model: {model_name} on device={self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def rerank(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of BM25 hits using cross-encoder scores.

        Args:
            query: user question
            hits: list of dicts with at least "text"
            top_k: number of passages to return

        Returns:
            Sorted subset of hits with added "rerank_score".
        """
        if not hits:
            return []

        texts = [h.get("text", "") for h in hits]
        pairs = [(query, t) for t in texts]

        enc = self.tokenizer(
            [q for q, _ in pairs],
            [p for _, p in pairs],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**enc)
        scores = outputs.logits.squeeze(-1)  # shape [N]
        scores = scores.detach().cpu().tolist()

        # Attach scores and sort
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)

        hits_sorted = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
        return hits_sorted[:top_k]
