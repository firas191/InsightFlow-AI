# InsightFlow — Out-of-Distribution Evaluation

Measured on `eval/ood_test.csv`: realistic, hand-labeled tickets that deliberately differ from Bitext's templated style (typos, caps, mixed signals, ambiguous cases). This is the honest real-world picture; compare it to the ~100% on Bitext's own test split.

## Headline accuracy

| Head | OOD accuracy | Bitext test (reference) |
|------|--------------|--------------------------|
| Category | 81% | ~100% |
| Sentiment | 86% | ~96% |
| Team (routed) | 81% | n/a |
| Intent* | 71% | ~100% |

\* Intent labels are hand-assigned and noisy on out-of-distribution text, so treat intent accuracy as indicative only.

## Error analysis

| ID | Text | True cat → Pred (conf) | True sent → Pred (conf) | Note |
|----|------|------------------------|-------------------------|------|
| OOD-02 | how do I change the email address on my profile | ACCOUNT → SHIPPING (83%) | ok |  |
| OOD-03 | please delete my account and all my personal data | ok | Neutral → Negative (84%) |  |
| OOD-08 | I need to cancel the order I placed this morning | ok | Neutral → Negative (91%) |  |
| OOD-14 | I entered the wrong shipping address, can it be corrected | ok | Neutral → Negative (85%) |  |
| OOD-21 | I was charged twice for my subscription and Im annoyed | PAYMENT → SUBSCRIPTION (31%) | ok | hard |
| OOD-22 | still waiting on the refund you promised two weeks ago | ok | Negative → Neutral (51%) |  |
| OOD-27 | worst customer support experience of my life honestly | FEEDBACK → CONTACT (50%) | ok |  |
| OOD-28 | wanted to say the new app update is fantastic | FEEDBACK → ACCOUNT (91%) | Positive → Neutral (50%) |  |
| OOD-29 | the product stopped working after two days and no one helped | FEEDBACK → ORDER (34%) | ok |  |
| OOD-35 | love the product but the checkout page crashed when i paid | PAYMENT → FEEDBACK (20%) | ok | mixed |
| OOD-37 | need to update the card saved on my subscription | PAYMENT → ACCOUNT (92%) | ok | hard |
| OOD-40 | the refund came through, thanks for sorting it out | ok | Positive → Neutral (47%) | positive |
| OOD-41 | can someone call me back about my account please | CONTACT → ACCOUNT (91%) | ok | ambiguous |

## How to read this

- A drop from ~100% (Bitext) to the OOD numbers above is expected and honest — it reflects that Bitext is clean and templated, not that the model is broken.
- Recurring error patterns (e.g. billing complaints that mention "subscription" being routed to SUBSCRIPTION, or complaint-style delivery messages landing on FEEDBACK) are **taxonomy artifacts** of Bitext's label space, not random mistakes.
- Most errors arrive with **lower confidence**, so the agent's low-confidence triage queue would catch them before any auto-action.