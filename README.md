# InsightFlow AI — Enterprise Pulse Agent

> Rapid LLM application built on the open-source ecosystem (HuggingFace Transformers + LangChain).
> This repo implements the **hardest, flagship module** from the project brief: the **Enterprise Pulse Agent**, an agent that turns scattered support tickets and feedback into a daily business "pulse" — *detect → understand → summarize → recommend → act*.

---

## What this is

The Enterprise Pulse Agent runs an end-to-end pipeline over incoming support tickets:

1. **Classify** each ticket (category, intent, sentiment) using a **fine-tuned multi-task transformer** — the trainable ML core of this project. The **routing team** is then derived from the category by rule.
2. **Retrieve** similar historical cases via embeddings + semantic search.
3. **Summarize** the day's top pain points (structured summary by default; optional distilBART abstractive layer).
4. **Draft** suggested responses for support agents.
5. **Report** a concise daily pulse for managers (risks + recommended actions), with **human-review checkpoints** before anything is acted on.

The classifier is trained on the **real [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)** (~26.9k rows) **on Kaggle** (GPU) via the notebook in `notebooks/`. `category` and `intent` use the dataset's real labels; `sentiment` is weak-labeled by a pretrained sentiment model at train time. Everything else uses pretrained open-source models and orchestration. The pipeline runs **immediately with a zero-shot fallback**, then gets faster and more accurate once you drop in the fine-tuned model.

---

## Repo structure

```
InsightFlow AI 2/
├── README.md                       # You are here
├── EVALUATION_REPORT.md            # Full evaluation write-up (metrics, OOD, limitations)
├── requirements.txt                # Python dependencies
├── config/
│   └── config.yaml                 # Label schemas, model names, paths (single source of truth)
├── data/
│   ├── sample_tickets.csv          # Demo input for the agent
│   ├── historical_cases.csv        # Knowledge base for retrieval / grounded drafts
│   └── generate_synthetic_data.py  # Legacy offline-demo data generator
├── notebooks/
│   └── train_ticket_classifier.ipynb   # >>> RUN THIS ON KAGGLE (GPU) <<<
├── src/
│   ├── config.py                   # Loads config.yaml
│   ├── classifier/
│   │   ├── model.py                # Multi-task model definition (shared encoder + heads)
│   │   └── inference.py            # Loads fine-tuned model; zero-shot fallback
│   ├── retriever.py                # Embedding + semantic search over historical cases
│   ├── summarizer.py               # Structured summary (+ optional distilBART)
│   ├── responder.py                # Draft response generation
│   ├── report.py                   # Daily pulse report builder
│   ├── tools.py                    # LangChain StructuredTools (search / ticket / export)
│   └── pulse_agent.py              # LangChain LCEL pipeline orchestration
├── eval/
│   ├── ood_test.csv                # Hand-labeled out-of-distribution test set
│   └── run_ood_eval.py             # Honest OOD evaluation -> reports/ood_eval.md
├── reports/                        # Generated daily pulse + evaluation outputs
├── models/
│   └── README.md                   # Where to drop the trained model from Kaggle
├── app/
│   └── streamlit_app.py            # Optional click-through demo UI
└── docs/
    └── architecture.md             # Workflow diagram + design notes
```

---

## Quickstart

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. (Optional) sample input for a demo
A small `data/sample_tickets.csv` is committed so you can run the agent without anything else. (The legacy `data/generate_synthetic_data.py` is kept only for an offline demo; **training uses the real Bitext dataset**, not synthetic data.)

### 3. Run the agent (works without training — uses zero-shot fallback)
```bash
python -m src.pulse_agent --input data/sample_tickets.csv
```
You'll get a printed daily pulse report. With no fine-tuned model present, classification falls back to a zero-shot model automatically.

### 4. Train the real classifier (on Kaggle)
Open `notebooks/train_ticket_classifier.ipynb` on Kaggle: **Add Input** → the *Bitext Gen AI Chatbot Customer Support* dataset; set **Internet: On** and **Accelerator: GPU**; Run All. It loads the real data, weak-labels sentiment, fine-tunes the multi-task transformer, evaluates, and writes a downloadable `insightflow_ticket_classifier.zip`. Download it, delete the old `models/insightflow_ticket_classifier/`, unzip the new one in its place, and re-run step 3 — the agent now reports `classifier mode: finetuned`.

### 5. (Optional) Click-through demo
```bash
streamlit run app/streamlit_app.py
```

### 6. Honest out-of-distribution evaluation
```bash
python eval/run_ood_eval.py
```
Bitext's own test split scores ~100% because it's clean and templated. This runs the model on `eval/ood_test.csv` — realistic, messy, hand-labeled tickets (typos, caps, ambiguous cases) — to reveal the *real-world* accuracy and an error-analysis table, written to `reports/ood_eval.md`. This is the core of the evaluation-report deliverable.

---

## The trainable core: multi-task ticket classifier

A single shared transformer encoder (DistilBERT by default; DeBERTa-v3-small optional) with **three classification heads** trained jointly on the real Bitext data:

| Head | Source | Classes |
|------|--------|---------|
| `category` | real Bitext label | 11: ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION |
| `intent` | real Bitext label | 27 fine-grained intents (cancel_order, get_refund, recover_password, …) |
| `sentiment` | weak-labeled by a pretrained sentiment model | Negative, Neutral, Positive |

`team` is **not** a learned head — it's routed from the predicted category by rule (`config.category_to_team`), which is how real systems do routing. `urgency` was dropped: the real data has no honest urgency label, and fabricating one reintroduces the shortcut-learning problem.

Multi-task learning lets one cheap encoder do three jobs at once — efficient for CPU inference at serve time, matching the brief's emphasis on latency and deployment constraints.

The label schema lives in `config/config.yaml` (single source of truth); the fine-tuned model also carries its own label order in `label_maps.json`.

---

## Responsible AI

This system is designed with the brief's ethics requirements built in:

- **Human review** — drafted responses and the daily report are *suggestions*; the report flags every item that needs human sign-off before action.
- **Uncertainty** — the classifier surfaces confidence scores; low-confidence tickets are flagged for manual triage rather than auto-routed.
- **Weak labels are disclosed** — sentiment is distilled from a teacher model, so it inherits that model's biases; this is called out as a limitation rather than presented as ground truth.
- **Privacy** — a PII-scrub hook redacts emails/phones before any model call; no real customer PII in the repo.
- **Safety** — the responder drafts are templated and constrained; nothing is sent automatically.

See `docs/architecture.md` for the full workflow diagram and human-in-the-loop checkpoints.

---

## License

Released under the [MIT License](LICENSE).

