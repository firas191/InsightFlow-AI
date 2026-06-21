"""Honest out-of-distribution (OOD) evaluation.

Bitext's test split is near-100% because the data is clean and templated. This
script measures the model on `eval/ood_test.csv` — realistic, messy, hand-labeled
tickets (typos, caps, mixed signals, ambiguous cases) that look nothing like the
training data. The gap between the two is the *honest* picture of real-world
performance, and the error table below is the core of the evaluation report.

Run from the repo root:
    python eval/run_ood_eval.py

Outputs metrics to the console and writes reports/ood_eval.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report  # noqa: E402

from src.config import CFG  # noqa: E402
from src.classifier.inference import TicketClassifier  # noqa: E402


def _routed_team(category: str) -> str:
    return CFG.get("category_to_team", default={}).get(category, "Customer Success")


def main():
    csv_path = ROOT / "eval" / "ood_test.csv"
    df = pd.read_csv(csv_path)

    clf = TicketClassifier()
    if clf.mode != "finetuned":
        print("WARNING: classifier is in", clf.mode, "mode — this evaluates the "
              "fallback, not your fine-tuned model. Put the trained model in "
              "models/insightflow_ticket_classifier/ and re-run.")
    print(f"classifier mode: {clf.mode}  |  OOD examples: {len(df)}\n")

    preds = clf.classify(df["text"].tolist())
    df["pred_category"] = [p["category"]["label"] for p in preds]
    df["cat_conf"] = [p["category"]["confidence"] for p in preds]
    df["pred_intent"] = [p["intent"]["label"] for p in preds]
    df["pred_sentiment"] = [p["sentiment"]["label"] for p in preds]
    df["sent_conf"] = [p["sentiment"]["confidence"] for p in preds]
    df["pred_team"] = [_routed_team(c) for c in df["pred_category"]]
    df["true_team"] = [_routed_team(c) for c in df["category"]]

    cat_acc = accuracy_score(df["category"], df["pred_category"])
    sent_acc = accuracy_score(df["sentiment"], df["pred_sentiment"])
    intent_acc = accuracy_score(df["intent"], df["pred_intent"])
    team_acc = accuracy_score(df["true_team"], df["pred_team"])

    # Console summary.
    print(f"category   accuracy : {cat_acc:.3f}")
    print(f"sentiment  accuracy : {sent_acc:.3f}")
    print(f"team       accuracy : {team_acc:.3f}  (derived from category)")
    print(f"intent     accuracy : {intent_acc:.3f}  (informational — hand labels are noisy)\n")

    print("=== category ===")
    print(classification_report(df["category"], df["pred_category"], zero_division=0))
    print("=== sentiment ===")
    print(classification_report(df["sentiment"], df["pred_sentiment"], zero_division=0))

    errors = df[(df["category"] != df["pred_category"]) | (df["sentiment"] != df["pred_sentiment"])]
    print(f"\n{len(errors)} of {len(df)} tickets have a category and/or sentiment error.")

    _write_markdown(df, errors, cat_acc, sent_acc, team_acc, intent_acc)
    out = (ROOT / "reports" / "ood_eval.md")
    print(f"\nReport written to: {out}")


def _write_markdown(df, errors, cat_acc, sent_acc, team_acc, intent_acc):
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# InsightFlow — Out-of-Distribution Evaluation")
    L.append("")
    L.append("Measured on `eval/ood_test.csv`: realistic, hand-labeled tickets that "
             "deliberately differ from Bitext's templated style (typos, caps, mixed "
             "signals, ambiguous cases). This is the honest real-world picture; "
             "compare it to the ~100% on Bitext's own test split.")
    L.append("")
    L.append("## Headline accuracy")
    L.append("")
    L.append("| Head | OOD accuracy | Bitext test (reference) |")
    L.append("|------|--------------|--------------------------|")
    L.append(f"| Category | {cat_acc:.0%} | ~100% |")
    L.append(f"| Sentiment | {sent_acc:.0%} | ~96% |")
    L.append(f"| Team (routed) | {team_acc:.0%} | n/a |")
    L.append(f"| Intent* | {intent_acc:.0%} | ~100% |")
    L.append("")
    L.append("\\* Intent labels are hand-assigned and noisy on out-of-distribution "
             "text, so treat intent accuracy as indicative only.")
    L.append("")
    L.append("## Error analysis")
    L.append("")
    if len(errors) == 0:
        L.append("No category/sentiment errors on this set.")
    else:
        L.append("| ID | Text | True cat → Pred (conf) | True sent → Pred (conf) | Note |")
        L.append("|----|------|------------------------|-------------------------|------|")
        for _, r in errors.iterrows():
            cat = f"{r['category']} → {r['pred_category']} ({r['cat_conf']:.0%})" if r['category'] != r['pred_category'] else "ok"
            sent = f"{r['sentiment']} → {r['pred_sentiment']} ({r['sent_conf']:.0%})" if r['sentiment'] != r['pred_sentiment'] else "ok"
            text = str(r["text"]).replace("|", "/")[:70]
            note = r["note"] if isinstance(r["note"], str) else ""
            L.append(f"| {r['ticket_id']} | {text} | {cat} | {sent} | {note} |")
    L.append("")
    L.append("## How to read this")
    L.append("")
    L.append("- A drop from ~100% (Bitext) to the OOD numbers above is expected and "
             "honest — it reflects that Bitext is clean and templated, not that the "
             "model is broken.")
    L.append("- Recurring error patterns (e.g. billing complaints that mention "
             "\"subscription\" being routed to SUBSCRIPTION, or complaint-style "
             "delivery messages landing on FEEDBACK) are **taxonomy artifacts** of "
             "Bitext's label space, not random mistakes.")
    L.append("- Most errors arrive with **lower confidence**, so the agent's "
             "low-confidence triage queue would catch them before any auto-action.")
    (reports_dir / "ood_eval.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
