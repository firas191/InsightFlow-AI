"""Semantic retrieval over historical resolved cases.

Encoder-model showcase: embeds the historical knowledge base with a
sentence-transformer and ranks cases by cosine similarity to the incoming
ticket. Pure-numpy search keeps the dependency surface small (no FAISS needed
for a few thousand cases).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG


class CaseRetriever:
    def __init__(self, cases_csv: str | Path | None = None, model_name: str | None = None):
        self.cases_csv = Path(cases_csv) if cases_csv else (CFG.root / "data" / "historical_cases.csv")
        self.model_name = model_name or CFG.models["embedder"]
        self._model = None
        self._cases: pd.DataFrame | None = None
        self._embeddings: np.ndarray | None = None

    def _ensure_loaded(self):
        if self._embeddings is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self._cases = pd.read_csv(self.cases_csv)
        # Embed "problem" text; normalize for cosine via dot product.
        corpus = self._cases["problem"].astype(str).tolist()
        emb = self._model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
        self._embeddings = emb.astype(np.float32)

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        self._ensure_loaded()
        k = k or CFG.get("agent", "retrieve_k", default=3)
        q = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = self._embeddings @ q  # cosine similarity (both normalized)
        top = np.argsort(-scores)[:k]
        out = []
        for idx in top:
            row = self._cases.iloc[int(idx)]
            out.append({
                "case_id": row["case_id"],
                "category": row["category"],
                "problem": row["problem"],
                "resolution": row["resolution"],
                "score": round(float(scores[idx]), 4),
            })
        return out
