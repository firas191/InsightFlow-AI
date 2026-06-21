# InsightFlow AI — Evaluation Report

**Project:** Enterprise Pulse Agent — a multi-task LLM application that turns unstructured support tickets into a daily business "pulse" (detect → understand → summarize → recommend → act).
**Date:** June 2026
**Author:** firas

---

## 1. Executive summary

InsightFlow AI classifies incoming customer-support tickets across three dimensions (category, intent, sentiment), routes each to the right team, retrieves similar resolved cases, drafts suggested replies, and compiles a manager-facing daily report — all with human-in-the-loop checkpoints.

The trainable core is a single shared **DistilBERT** encoder with three classification heads, fine-tuned on the **real Bitext customer-support dataset** (~26.9k examples). The headline finding of this evaluation is a deliberate one: on the clean, templated Bitext test split the model scores ~100%, but on a hand-built **out-of-distribution (OOD)** set of realistic, messy tickets it scores **81% (category)** and **86% (sentiment)**. That gap is the honest measure of real-world performance, and characterising it — rather than reporting the inflated in-distribution number — is the central contribution of this evaluation.

A three-round tuning process is documented in full, including a round that **regressed** performance, because the failed experiment is as informative as the successful ones.

---

## 2. What was built

| Component | Implementation | Course topic demonstrated |
|---|---|---|
| Multi-task classifier (trainable) | DistilBERT + 3 linear heads, fine-tuned on Kaggle GPU | Encoder models, fine-tuning, transfer learning |
| Sentiment labels | Distilled from a pretrained teacher + targeted augmentation | Weak supervision, class imbalance |
| Retrieval | Sentence-Transformers (MiniLM) embeddings + cosine search over historical cases | Embeddings, semantic search |
| Summarisation | Structured summary by default; optional distilBART (encoder-decoder) abstractive layer | Decoder / encoder-decoder summarisation |
| Response drafting | Template grounded in the closest retrieved resolution | Safe, controllable text generation |
| Orchestration + tools | LangChain LCEL Runnable pipeline; side-effects via LangChain `StructuredTool.invoke` | Multi-step workflows, agentic tool use |
| Routing & escalation | Rule-based team routing + sentiment/priority escalation | Connecting NLP to business logic |

The pipeline runs immediately with a **zero-shot fallback** when no fine-tuned model is present, and switches to the fine-tuned model once it is installed.

---

## 3. Data and model

**Dataset.** The Bitext "Customer Support" dataset (~26,872 rows) was chosen after rejecting a more obvious candidate. The widely used "Customer Support Ticket Dataset" (Suraj, ~8.5k rows) was inspected and found unusable: its labels are effectively random relative to the ticket text, and the descriptions contain unfilled `{product_purchased}` placeholders. Bitext, by contrast, has genuine, well-correlated `category` (11 classes) and `intent` (27 classes) labels.

**Label schema.**
- `category` (11): ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION — **real** Bitext labels.
- `intent` (27): cancel_order, get_refund, recover_password, … — **real** Bitext labels.
- `sentiment` (3): Negative, Neutral, Positive — **weak-labeled** (see below).
- `team`: **not learned** — routed from category by rule (`config.category_to_team`), as real support desks do.
- `urgency`: **dropped** — Bitext has no honest urgency label, and fabricating one reintroduces shortcut learning (see §4).

**Model exploration (≥3 models considered).**

| Model | Role | Why |
|---|---|---|
| `distilbert-base-uncased` (66M) | Chosen encoder | Fast CPU inference, strong accuracy/size trade-off |
| `microsoft/deberta-v3-small` | Considered alternative | Higher accuracy at more compute; flagged for future work |
| `facebook/bart-large-mnli` | Zero-shot fallback | Runs the pipeline with no fine-tuned model |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment teacher | Distils weak sentiment labels |
| `sentence-transformers/all-MiniLM-L6-v2` | Retrieval embeddings | Lightweight semantic search |
| `sshleifer/distilbart-cnn-12-6` | Summariser | Distilled encoder-decoder for abstractive summary |

---

## 4. Methodology and the iteration story

### 4.1 The synthetic baseline (rejected)

An initial version trained on a synthetic, templated dataset scored **100% on every head** on its own test split. This was a red flag, not a success: the generator embedded each label as a fixed phrase (sentiment from the opener, urgency from the tail), so the model learned phrase→label **shortcuts**. It failed on any text outside the template. This motivated the switch to real Bitext data.

### 4.2 Sentiment imbalance and the fix

The sentiment teacher labeled only **13 of 26,872** tickets as Positive — support tickets are rarely positive. The first Bitext model therefore had a dead Positive class (macro-F1 0.63). The fix was twofold:
1. **Augmentation:** ~700 varied positive *reviews* added, labeled `FEEDBACK`/`review` so they stay consistent with the category and intent heads.
2. **Class weights:** inverse-frequency weights on the sentiment loss.

This lifted in-distribution sentiment macro-F1 from **0.63 → 0.97**.

### 4.3 Three tuning rounds (OOD-measured)

| Round | Changes | Category (OOD) | Sentiment (OOD) | Negative recall | Positive precision | Angry→Positive flips |
|---|---|---|---|---|---|---|
| v1 | Bitext + balanced sentiment | 81% | 83% | 0.93 | 1.00 | 0 |
| v2 | + broad augmentation, label smoothing, text augmentation | 79% | **74%** | 0.80 | **0.36** | **2** |
| **v3 (final)** | calibration kept, augmentation reverted, class weight capped | **81%** | **86%** | **0.93** | **1.00** | **0** |

**v2 was a regression** and is reported honestly. Broadening the positive vocabulary with transactional words ("the delivery arrived early", "the checkout was quick") leaked domain terms into the Positive class; combined with a ~10× class weight, the model over-predicted Positive (precision collapsed to 0.36) and **flipped two angry customers to Positive** — the unsafe direction. The only genuine win from v2 was **label smoothing**, which lowered overconfidence.

**v3** kept the calibration win and reverted the harm: domain-neutral positive vocabulary, Positive class weight capped at 5×, label smoothing retained. It is the best version on every axis.

---

## 5. Results

### 5.1 In-distribution (Bitext test split)

| Head | Accuracy | Macro-F1 |
|---|---|---|
| Category | ~100% | ~1.00 |
| Intent (27-class) | ~100% | ~1.00 |
| Sentiment | ~96% | ~0.97 |

These numbers are high **because Bitext is clean and templated**, not because the model is robust. They are reported only as a reference point.

### 5.2 Out-of-distribution (the honest result)

Measured on `eval/ood_test.csv` — 42 hand-labeled tickets with typos, caps, mixed signals, and ambiguous cases that look nothing like Bitext.

| Head | OOD accuracy |
|---|---|
| Category | **81%** |
| Sentiment | **86%** |
| Team (routed) | 81% |
| Intent (informational) | 71% |

Sentiment, per class (OOD): Negative F1 0.88 (recall **0.93** — strong, and the safety-critical direction), Neutral F1 0.87, Positive F1 0.67 (recall 0.50 — see limitations).

### 5.3 Error analysis (four recurring patterns)

1. **Keyword-shortcut misroutes (the main weakness).** A few high-confidence category errors where a salient token dominates: "change the email on my profile" → SHIPPING, "update the card on my subscription" → ACCOUNT. Bitext tied these tokens tightly to one intent.
2. **Taxonomy-overlap errors (self-catching).** FEEDBACK vs CONTACT vs PAYMENT on complaint/mixed messages, arriving at 20–50% confidence — caught by the triage queue.
3. **Sentiment over-flags negativity (safe direction).** Neutral requests with "cancel", "delete", "wrong" marked Negative → harmless over-escalation.
4. **Positive recall 0.50 (disclosed cost).** The model misses subtler positives ("the update is fantastic"); this is the direct, expected consequence of the partly-synthetic Positive class.

Label smoothing measurably improved calibration: across runs the hardest errors dropped from 60–90% confidence into the 20–50% range, i.e. into the triage queue rather than auto-action.

---

## 6. End-to-end agent demonstration

Running the full agent over 28 sample tickets produced a coherent daily pulse:

- **28 tickets processed, 15 escalated, 7 flagged for human triage.**
- Executive summary (structured): *"28 tickets this batch, 43% negative. Negative sentiment clusters around order (3), refund (2), delivery (1)."*
- Breakdown by category, sentiment, routed team, and top intents.
- The triage queue correctly caught the genuinely confused tickets at low confidence — e.g. "the product arrived broken and nobody has responded" at **21%**.
- Drafted replies were grounded in the right historical cases: the refund ticket cited the refund-recovery case, the double-charge ticket cited the duplicate-charge reversal case, the missing-package ticket cited the lost-in-transit case.

Notably, the double-charge ticket was mis-categorised as ORDER (the word "order" dominated), yet the retriever **still** surfaced the correct duplicate-charge resolution — the pipeline degraded gracefully.

---

## 7. Latency and cost

The model is intentionally small (DistilBERT, 66M parameters) and produces all three predictions in a **single forward pass**, which suits CPU serving. In practice the 42-example OOD evaluation and the 28-ticket agent run both completed classification in well under a second of compute on CPU. The only heavy component is the optional abstractive summariser (distilBART, ~1.2 GB), which is not required — the structured summary covers the same need at negligible cost. A formal millisecond-level latency benchmark is listed under future work.

---

## 8. Limitations

- **In-distribution scores overstate real performance.** Bitext is clean and templated; the OOD set is the trustworthy measure (~81% category).
- **Sentiment is weak-labeled and partly synthetic.** The Positive class is augmented, so its real-world reliability is below the reported figure; the head inherits the teacher model's biases.
- **Taxonomy artifacts.** Bitext's label space does not cleanly separate ACCOUNT/CONTACT or PAYMENT/SUBSCRIPTION, producing a few unavoidable confusions.
- **Over-escalation.** The priority-category rule escalates every payment/refund/cancel/contact ticket regardless of sentiment, inflating the escalation count (a deliberate "safe" bias, but tunable).
- **No urgency dimension** — dropped for lack of an honest label.
- **Intent OOD accuracy (71%) is indicative only** — hand-labeling 27 intents on novel text is noisy.

---

## 9. Responsible AI

- **Human-in-the-loop by design.** Low-confidence predictions go to a triage queue; every drafted reply is labeled `[DRAFT — review before sending]`; follow-up actions are gated on human approval. Nothing is sent automatically.
- **Calibrated confidence.** Label smoothing keeps the model from being falsely certain, so more of its mistakes surface for human review.
- **Privacy.** A PII-scrub step redacts emails and phone numbers before any model call; the repository contains no real customer data.
- **Honesty about weak labels.** The sentiment labels are distilled/augmented and are disclosed as such rather than presented as ground truth.
- **Safe generation.** Drafted replies are templated and grounded in real past resolutions, avoiding unsupported or hallucinated claims.

---

## 10. Improvement ideas

1. **Replace weak/synthetic sentiment with real labeled sentiment data** to remove the Positive-class caveat.
2. **Swap DistilBERT for DeBERTa-v3-small** for better out-of-distribution generalisation.
3. **Refine the label taxonomy** (merge or disambiguate overlapping categories) to cut the keyword-shortcut errors.
4. **Soften escalation** — require negative sentiment for priority categories, or weight by confidence.
5. **Temperature-scale on a held-out OOD set** for sharper calibration of the triage gate.
6. **Expand the OOD evaluation set**, ideally with real anonymised tickets, and add a formal latency benchmark.

---

## 11. Conclusion

InsightFlow AI demonstrates a complete, honest LLM application: real-data fine-tuning, multi-task classification, embeddings-based retrieval, summarisation, grounded generation, and agentic orchestration with human oversight. The most valuable outcome is not the ~100% in-distribution score but the **measured ~81–86% out-of-distribution performance**, the documented error patterns, and a tuning process that includes a failed round. A model that knows where it is weak — and routes its weak cases to a human — is more useful in production than one that merely reports a high number on clean data.
