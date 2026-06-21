"""Executive pain-point summary for the daily pulse.

Design: the summary is built from the *structured* predictions (counts,
sentiment, dominant negative themes), so it is always fluent and informative.
A generative encoder-decoder model (distilBART) is attempted as an optional
enhancement over the negative tickets — if it loads, an abstractive sentence is
appended; if it errors, we log why and the structured summary stands alone.

Note: we call the seq2seq model directly via `AutoModelForSeq2SeqLM.generate`
rather than `pipeline("summarization")`, because transformers 5.0 removed the
"summarization" pipeline task alias.
"""
from __future__ import annotations

import sys
from collections import Counter

from src.config import CFG


class PainPointSummarizer:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or CFG.models["summarizer"]
        self._tok = None
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def summarize(self, processed: list[dict], max_words: int = 70) -> str:
        if not processed:
            return "No tickets to summarize in this batch."

        n = len(processed)
        sentiments = Counter(p["prediction"]["sentiment"]["label"] for p in processed)
        pct_neg = sentiments.get("Negative", 0) / n
        negatives = [p for p in processed if p["prediction"]["sentiment"]["label"] == "Negative"]
        neg_cats = Counter(p["prediction"]["category"]["label"] for p in negatives)

        if neg_cats:
            themes = ", ".join(f"{cat.lower()} ({cnt})" for cat, cnt in neg_cats.most_common(3))
            base = (f"{n} tickets this batch, {pct_neg:.0%} negative. "
                    f"Negative sentiment clusters around {themes}.")
        else:
            base = f"{n} tickets this batch, {pct_neg:.0%} negative — no dominant negative theme."

        abstractive = ""
        if CFG.get("agent", "use_generative_summary", default=False):
            abstractive = self._try_abstractive([p["text"] for p in negatives], max_words)
        return f"{base} {abstractive}".strip() if abstractive else base

    def _try_abstractive(self, texts: list[str], max_words: int) -> str:
        texts = [t for t in texts if t and t.strip()]
        if len(texts) < 3:
            return ""
        joined = " ".join(" ".join(texts).split()[:600])
        try:
            import torch

            self._ensure_model()
            inputs = self._tok(joined, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                out_ids = self._model.generate(
                    **inputs,
                    max_length=max(40, max_words),
                    min_length=20,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                )
            return self._tok.decode(out_ids[0], skip_special_tokens=True).strip()
        except Exception as exc:  # surfaced, not swallowed
            print(f"[summarizer] generative summary unavailable "
                  f"({type(exc).__name__}: {exc}); using structured summary.", file=sys.stderr)
            return ""
