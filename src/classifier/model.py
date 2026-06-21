"""Multi-task transformer classifier.

One shared encoder (DistilBERT / DeBERTa / any HF encoder) feeds several
independent linear heads — one per task (category, urgency, team, sentiment).
Training jointly optimizes all heads; at serve time one cheap forward pass
produces all four predictions, which is friendly to CPU latency budgets.

This module is imported BOTH by the Kaggle training notebook and by
src/classifier/inference.py, so the architecture and the save/load format are
guaranteed identical on both sides.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

HEADS_FILE = "heads.pt"
META_FILE = "label_maps.json"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-masked mean pooling — robust across encoders (no pooler needed)."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class MultiTaskClassifier(nn.Module):
    def __init__(self, base_model_name: str, head_labels: dict[str, list[str]], dropout: float = 0.1):
        super().__init__()
        self.base_model_name = base_model_name
        self.head_labels = head_labels  # {task: [label, ...]} — index == class id
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(hidden, len(labels)) for task, labels in head_labels.items()}
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        return {task: head(pooled) for task, head in self.heads.items()}

    # ---- persistence -------------------------------------------------------
    def save(self, out_dir: str | Path, tokenizer: AutoTokenizer | None = None) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Encoder weights + config (standard HF format).
        self.encoder.save_pretrained(out_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)
        # Heads + metadata.
        torch.save({t: h.state_dict() for t, h in self.heads.items()}, out_dir / HEADS_FILE)
        with open(out_dir / META_FILE, "w", encoding="utf-8") as fh:
            json.dump(
                {"base_model_name": self.base_model_name, "head_labels": self.head_labels},
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cpu") -> "MultiTaskClassifier":
        model_dir = Path(model_dir)
        with open(model_dir / META_FILE, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        # Build the skeleton, then load the encoder weights from the saved dir
        # (passing model_dir makes AutoModel read the fine-tuned weights).
        model = cls(str(model_dir), meta["head_labels"])
        model.base_model_name = meta["base_model_name"]
        head_state = torch.load(model_dir / HEADS_FILE, map_location=device)
        for task, sd in head_state.items():
            model.heads[task].load_state_dict(sd)
        model.to(device).eval()
        return model
