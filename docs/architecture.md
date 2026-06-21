# InsightFlow AI — Architecture & Workflow

The Enterprise Pulse Agent turns a stream of support tickets into a daily
business "pulse": *detect → understand → summarize → recommend → act*, with a
human in the loop before anything is acted on.

## Workflow diagram

```mermaid
flowchart TD
    A[Incoming tickets<br/>CSV / feed] --> B[PII scrub<br/>redact emails & phones]
    B --> C{Multi-task classifier}
    C -->|fine-tuned model present| C1[DistilBERT + 3 heads<br/>category · intent · sentiment]
    C -->|no model yet| C2[Zero-shot fallback<br/>bart-large-mnli]
    C1 --> D[Per-ticket predictions + confidence<br/>team routed from category]
    C2 --> D

    D --> E{Needs attention?<br/>negative sentiment OR priority category}
    E -->|yes| F[Embed + semantic search<br/>MiniLM over historical cases]
    F --> G[Draft suggested response<br/>grounded in closest case]
    E -->|no| H[Skip drafting]

    D --> I{Low confidence?<br/>below threshold}
    I -->|yes| J[(Human triage queue)]
    I -->|no| K[Auto-route to team]

    G --> L[Summarize pain points<br/>distilBART]
    H --> L
    K --> L
    J --> L
    L --> M[Build Daily Pulse report<br/>risks · actions · drafts]
    M --> N[[export_report tool]]
    M --> O{{Human review}}
    O -->|approve| P[create_followup_ticket tool<br/>+ send drafted responses]

    classDef human fill:#fde68a,stroke:#b45309,color:#000;
    classDef tool fill:#bfdbfe,stroke:#1e40af,color:#000;
    class J,O human;
    class N,P tool;
```

## Components → brief mapping

| Component | File | Brief topic |
|-----------|------|-------------|
| Multi-task classifier (trainable) | `notebooks/train_ticket_classifier.ipynb`, `src/classifier/` | Encoder models, fine-tuning, zero-shot classification |
| Semantic retrieval | `src/retriever.py` | Embeddings, semantic search, Knowledge Q&A |
| Pain-point summarizer | `src/summarizer.py` | Decoder / encoder-decoder summarization |
| Response drafter | `src/responder.py` | Scalable text generation (safe, templated) |
| Daily Pulse report | `src/report.py` | Report Generator module |
| Tools (search / ticket / export) | `src/tools.py` | LangChain agents + tool use |
| Orchestration | `src/pulse_agent.py` | LangChain-style multi-step workflow |

## Human-in-the-loop checkpoints

1. **Low-confidence triage** — any ticket whose category/sentiment confidence
   falls below `agent.low_confidence_threshold` (config) is queued for a human
   instead of auto-routed.
2. **Report review** — the daily report explicitly labels every drafted
   response a `[DRAFT — review before sending]` and lists escalations and
   recommended actions as *suggestions*.
3. **Action gate** — `create_followup_ticket` and sending responses only happen
   after human approval; nothing is dispatched automatically.

## Responsible-AI hooks

- **Privacy:** `scrub_pii()` redacts emails/phones before any model call.
- **Uncertainty:** confidence scores are surfaced and drive the triage queue.
- **Weak labels disclosed:** sentiment is distilled from a teacher model and
  inherits its biases — flagged as a limitation, not treated as ground truth.
- **Safety:** responses are template-grounded in real past resolutions, not
  free-form generation; nothing is auto-sent.
- **Cost/latency:** one small shared encoder serves three tasks per forward pass,
  matching CPU-serving constraints; the heavy training runs offline on Kaggle.

## Data flow contract

The label schema in `config/config.yaml` is the single source of truth, and the
fine-tuned model additionally carries its own label order in `label_maps.json`,
so a model trained on Kaggle loads and predicts correctly with no glue code.
`inference.py` verifies the saved heads match the configured `learned_heads`
(category / intent / sentiment) and falls back to zero-shot on a mismatch. The
saved artifact layout is:

```
insightflow_ticket_classifier/
├── config.json            # encoder architecture (HF)
├── model.safetensors      # fine-tuned encoder weights
├── tokenizer files
├── heads.pt               # {task: linear-head state_dict}
└── label_maps.json        # {base_model_name, head_labels}
```
