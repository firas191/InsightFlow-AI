"""Ticket classification at serve time.

Two modes, chosen automatically:
  * "finetuned"  — loads the multi-task model trained on Kaggle (fast, accurate).
  * "zero-shot"  — falls back to a zero-shot NLI model when no fine-tuned
                   artifact is present, so the whole pipeline runs out of the box.

Public API:
    clf = TicketClassifier()          # auto-detects mode
    preds = clf.classify([text, ...]) # -> list of per-ticket prediction dicts
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.config import CFG

# Learned heads predicted by the model.
TASKS = ["category", "intent", "sentiment"]
# Heads that gate auto-routing (intent is fine-grained, so excluded from triage).
DECISION_TASKS = ["category", "sentiment"]


class TicketClassifier:
    def __init__(self, device: str | None = None, prefer_finetuned: bool = True):
        import torch  # local import keeps module import cheap

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = None
        self._model = None
        self._tokenizer = None
        self._zs = None  # zero-shot pipeline (lazy)

        model_dir = CFG.path("models", "classifier_dir") if "classifier_dir" in CFG.models else None
        meta_ok = model_dir is not None and (Path(model_dir) / "label_maps.json").exists()

        if prefer_finetuned and meta_ok and self._schema_matches(model_dir):
            self._load_finetuned(model_dir)
        else:
            if prefer_finetuned and meta_ok:
                print("[classifier] fine-tuned model heads do not match the current schema "
                      f"({CFG.get('learned_heads', default=TASKS)}). Falling back to zero-shot. "
                      "Retrain with the current notebook and replace models/insightflow_ticket_classifier/.")
            self.mode = "zero-shot"

    @staticmethod
    def _schema_matches(model_dir) -> bool:
        import json

        try:
            with open(Path(model_dir) / "label_maps.json", encoding="utf-8") as fh:
                heads = set(json.load(fh).get("head_labels", {}).keys())
        except Exception:
            return False
        expected = set(CFG.get("learned_heads", default=TASKS))
        return heads == expected

    # ---- loading -----------------------------------------------------------
    def _load_finetuned(self, model_dir):
        from transformers import AutoTokenizer

        from src.classifier.model import MultiTaskClassifier

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = MultiTaskClassifier.load(model_dir, device=self.device)
        self.mode = "finetuned"

    def _ensure_zero_shot(self):
        if self._zs is None:
            from transformers import pipeline

            self._zs = pipeline(
                "zero-shot-classification",
                model=CFG.models["zero_shot"],
                device=0 if self.device == "cuda" else -1,
            )

    # ---- prediction --------------------------------------------------------
    def classify(self, texts: Iterable[str], batch_size: int = 16) -> list[dict]:
        texts = list(texts)
        if not texts:
            return []
        if self.mode == "finetuned":
            return self._classify_finetuned(texts, batch_size)
        return self._classify_zero_shot(texts)

    def _classify_finetuned(self, texts, batch_size) -> list[dict]:
        torch = self.torch
        max_len = CFG.get("training", "max_length", default=192)
        results: list[dict] = []
        self._model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self._tokenizer(
                    batch, truncation=True, padding=True,
                    max_length=max_len, return_tensors="pt",
                ).to(self.device)
                logits = self._model(enc["input_ids"], enc["attention_mask"])
                # Per-task softmax -> top label + confidence.
                for row in range(len(batch)):
                    pred = {"_mode": "finetuned"}
                    for task in self._model.head_labels:
                        probs = torch.softmax(logits[task][row], dim=-1)
                        conf, idx = torch.max(probs, dim=-1)
                        pred[task] = {
                            "label": self._model.head_labels[task][int(idx)],
                            "confidence": round(float(conf), 4),
                        }
                    results.append(pred)
        return results

    def _classify_zero_shot(self, texts) -> list[dict]:
        self._ensure_zero_shot()
        results = []
        for text in texts:
            pred = {"_mode": "zero-shot"}
            for task in TASKS:
                labels = CFG.heads[task]
                out = self._zs(text, candidate_labels=labels, multi_label=False)
                pred[task] = {
                    "label": out["labels"][0],
                    "confidence": round(float(out["scores"][0]), 4),
                }
            results.append(pred)
        return results

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def min_confidence(pred: dict) -> float:
        vals = [pred[t]["confidence"] for t in DECISION_TASKS if t in pred]
        return min(vals) if vals else 1.0
