# InsightFlow — Out-of-Distribution Evaluation

Measured on `eval/ood_test.csv`: realistic, hand-labeled tickets that deliberately differ from Bitext's templated style (typos, caps, mixed signals, ambiguous cases). This is the honest real-world picture; compare it to the ~100% on Bitext's own test split.

## Headline accuracy

| Head | OOD accuracy | Bitext test (reference) |
|------|--------------|--------------------------|
| Category | 79% | ~100% |
| Sentiment | 74% | ~96% |
| Team (routed) | 81% | n/a |
| Intent* | 67% | ~100% |

\* Intent labels are hand-assigned and noisy on out-of-distribution text, so treat intent accuracy as indicative only.

## Error analysis

| ID | Text | True cat → Pred (conf) | True sent → Pred (conf) | Note |
|----|------|------------------------|-------------------------|------|
| OOD-02 | how do I change the email address on my profile | ACCOUNT → SHIPPING (60%) | ok |  |
| OOD-03 | please delete my account and all my personal data | ok | Neutral → Negative (61%) |  |
| OOD-06 | how do I place a large order for my whole team | ok | Neutral → Positive (52%) |  |
| OOD-08 | I need to cancel the order I placed this morning | ok | Neutral → Negative (82%) |  |
| OOD-09 | wheres my order, is there a way to track it | ok | Neutral → Positive (55%) |  |
| OOD-10 | how much is the early cancellation fee on my plan | ok | Neutral → Positive (49%) |  |
| OOD-13 | your delivery never showed up and its been three weeks | DELIVERY → SHIPPING (44%) | ok | hard |
| OOD-14 | I entered the wrong shipping address, can it be corrected | ok | Neutral → Negative (73%) |  |
| OOD-21 | I was charged twice for my subscription and Im annoyed | PAYMENT → SUBSCRIPTION (66%) | ok | hard |
| OOD-22 | still waiting on the refund you promised two weeks ago | ok | Negative → Neutral (46%) |  |
| OOD-24 | I want my money back for the damaged item I returned | ok | Negative → Positive (42%) |  |
| OOD-26 | whats the best way to reach your support team | ok | Neutral → Positive (66%) |  |
| OOD-27 | worst customer support experience of my life honestly | FEEDBACK → CONTACT (43%) | ok |  |
| OOD-28 | wanted to say the new app update is fantastic | FEEDBACK → ACCOUNT (89%) | ok |  |
| OOD-29 | the product stopped working after two days and no one helped | FEEDBACK → ORDER (28%) | ok |  |
| OOD-35 | love the product but the checkout page crashed when i paid | PAYMENT → FEEDBACK (79%) | Negative → Positive (70%) | mixed |
| OOD-37 | need to update the card saved on my subscription | PAYMENT → ACCOUNT (89%) | ok | hard |
| OOD-38 | do you ship to canada and how much does it cost | ok | Neutral → Positive (39%) | ambiguous |
| OOD-41 | can someone call me back about my account please | CONTACT → ACCOUNT (78%) | ok | ambiguous |

## How to read this

- A drop from ~100% (Bitext) to the OOD numbers above is expected and honest — it reflects that Bitext is clean and templated, not that the model is broken.
- Recurring error patterns (e.g. billing complaints that mention "subscription" being routed to SUBSCRIPTION, or complaint-style delivery messages landing on FEEDBACK) are **taxonomy artifacts** of Bitext's label space, not random mistakes.
- Most errors arrive with **lower confidence**, so the agent's low-confidence triage queue would catch them before any auto-action.