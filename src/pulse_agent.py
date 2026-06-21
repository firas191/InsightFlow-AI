"""Enterprise Pulse Agent — end-to-end orchestration.

Pipeline: ingest -> PII scrub -> classify -> retrieve similar cases ->
draft responses -> summarize pain points -> build & export daily report.

This is the deterministic backbone of the agent. The same capabilities are
also exposed as LangChain tools in src/tools.py for an LLM-driven agent.

CLI:
    python -m src.pulse_agent --input data/sample_tickets.csv
    python -m src.pulse_agent --input data/sample_tickets.csv --limit 50 --output today.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from src.config import CFG

# Redaction patterns for the privacy hook.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b")


def scrub_pii(text: str) -> str:
    """Privacy hook: redact obvious emails/phone numbers before model calls."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    return text


def load_tickets(path: str | Path, limit: int | None = None) -> list[dict]:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must have a 'text' column.")
    if "ticket_id" not in df.columns:
        df.insert(0, "ticket_id", [f"TK-{i:05d}" for i in range(len(df))])
    if limit:
        df = df.head(limit)
    return df[["ticket_id", "text"]].to_dict("records")


def run_pipeline(tickets: list[dict], prefer_finetuned: bool = True,
                 export: bool = True, output: str | None = None) -> dict:
    # Lazy imports so `--help` and unit tests don't load heavy models.
    from src.classifier.inference import TicketClassifier
    from src.responder import ResponseDrafter
    from src.retriever import CaseRetriever
    from src.summarizer import PainPointSummarizer
    from src import report as report_mod
    from src import tools

    thr = CFG.get("agent", "low_confidence_threshold", default=0.55)
    priority_categories = set(CFG.get("agent", "priority_categories", default=[]))
    escalate_neg = CFG.get("agent", "escalate_on_negative_sentiment", default=True)
    routing = CFG.get("category_to_team", default={})

    # 1) Privacy scrub.
    texts = [scrub_pii(t["text"]) for t in tickets]

    # 2) Classify (multi-task).
    clf = TicketClassifier(prefer_finetuned=prefer_finetuned)
    print(f"[agent] classifier mode: {clf.mode}  |  tickets: {len(texts)}", file=sys.stderr)
    preds = clf.classify(texts)

    # 3) Retrieve + 4) draft for escalated/negative tickets.
    retriever = CaseRetriever()
    drafter = ResponseDrafter()
    processed: list[dict] = []

    for tk, text, pred in zip(tickets, texts, preds):
        category = pred["category"]["label"]
        sentiment = pred["sentiment"]["label"]
        # Rule-based team routing derived from the predicted category.
        pred["team"] = {"label": routing.get(category, "Customer Success"),
                        "confidence": pred["category"]["confidence"]}
        low_conf = TicketClassifier.min_confidence(pred) < thr
        needs_attention = (escalate_neg and sentiment == "Negative") or category in priority_categories

        similar, draft = [], None
        if needs_attention:
            similar = retriever.retrieve(text)
            draft = drafter.draft(text, pred, similar)

        processed.append({
            "ticket_id": tk["ticket_id"],
            "text": text,
            "prediction": pred,
            "similar_cases": similar,
            "draft": draft,
            "low_confidence": low_conf,
            "escalated": needs_attention,
        })

    # 5) Summarize the day's pain points (structured + optional abstractive).
    summarizer = PainPointSummarizer()
    summary = summarizer.summarize(processed)

    # 6) Build the report; optionally export via the agent's export tool.
    markdown = report_mod.build_report(processed, summary)
    report_path = tools.export_report(markdown, filename=output) if export else None

    return {"report": markdown, "report_path": report_path,
            "processed": processed, "summary": summary, "mode": clf.mode}


def main():
    ap = argparse.ArgumentParser(description="Run the Enterprise Pulse Agent over a ticket CSV.")
    ap.add_argument("--input", default=str(CFG.get("paths", "sample_csv", default="data/sample_tickets.csv")),
                    help="path to a CSV with a 'text' column")
    ap.add_argument("--output", default=None, help="report filename (written to reports/)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N tickets")
    ap.add_argument("--no-finetuned", action="store_true",
                    help="force zero-shot fallback even if a fine-tuned model exists")
    args = ap.parse_args()

    tickets = load_tickets(args.input, args.limit)
    result = run_pipeline(tickets, prefer_finetuned=not args.no_finetuned, output=args.output)

    print(result["report"])
    if result["report_path"]:
        print(f"\n[agent] report saved to: {result['report_path']}", file=sys.stderr)


if __name__ == "__main__":
    main()
