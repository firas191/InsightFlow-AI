"""Enterprise Pulse Agent — end-to-end orchestration.

Pipeline: ingest -> PII scrub -> classify -> retrieve similar cases ->
draft responses -> summarize pain points -> build & export daily report.

Orchestration is built with **LangChain**: the four stages are composed as
LangChain Runnables (LCEL `a | b | c | d`) and invoked with `.invoke()`, and
the side-effecting capabilities (knowledge search, report export) are called
through the LangChain `StructuredTool` interface (`tool.invoke(...)`). If
LangChain isn't installed, the same step functions run as a plain sequence.

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
    from src.summarizer import PainPointSummarizer
    from src import report as report_mod
    from src import tools

    thr = CFG.get("agent", "low_confidence_threshold", default=0.55)
    priority_categories = set(CFG.get("agent", "priority_categories", default=[]))
    escalate_neg = CFG.get("agent", "escalate_on_negative_sentiment", default=True)
    routing = CFG.get("category_to_team", default={})

    clf = TicketClassifier(prefer_finetuned=prefer_finetuned)
    drafter = ResponseDrafter()
    summarizer = PainPointSummarizer()
    tool_map = tools.langchain_tool_map()  # name -> LangChain StructuredTool (or {})

    def knowledge_search(text: str):
        tool = tool_map.get("knowledge_search")
        return tool.invoke({"query": text}) if tool else tools.knowledge_search(text)

    def export_report(markdown: str, filename: str | None):
        tool = tool_map.get("export_report")
        if tool:
            return tool.invoke({"markdown": markdown, "filename": filename})
        return tools.export_report(markdown, filename)

    # ---- Pipeline stages, each a state-dict -> state-dict function ----
    def classify_stage(state: dict) -> dict:
        texts = [scrub_pii(t["text"]) for t in state["tickets"]]
        preds = clf.classify(texts)
        processed = []
        for tk, text, pred in zip(state["tickets"], texts, preds):
            category = pred["category"]["label"]
            sentiment = pred["sentiment"]["label"]
            pred["team"] = {"label": routing.get(category, "Customer Success"),
                            "confidence": pred["category"]["confidence"]}
            escalated = (escalate_neg and sentiment == "Negative") or category in priority_categories
            processed.append({
                "ticket_id": tk["ticket_id"], "text": text, "prediction": pred,
                "similar_cases": [], "draft": None,
                "low_confidence": TicketClassifier.min_confidence(pred) < thr,
                "escalated": escalated,
            })
        state["processed"] = processed
        return state

    def enrich_stage(state: dict) -> dict:
        for p in state["processed"]:
            if p["escalated"]:
                p["similar_cases"] = knowledge_search(p["text"])  # via LangChain tool
                p["draft"] = drafter.draft(p["text"], p["prediction"], p["similar_cases"])
        return state

    def summarize_stage(state: dict) -> dict:
        state["summary"] = summarizer.summarize(state["processed"])
        return state

    def report_stage(state: dict) -> dict:
        markdown = report_mod.build_report(state["processed"], state["summary"])
        state["report"] = markdown
        state["report_path"] = export_report(markdown, output) if export else None  # via LangChain tool
        return state

    # ---- Compose + run the stages via LangChain (fallback: plain sequence) ----
    orchestrator = "python-sequence"
    try:
        from langchain_core.runnables import RunnableLambda

        chain = (RunnableLambda(classify_stage) | RunnableLambda(enrich_stage)
                 | RunnableLambda(summarize_stage) | RunnableLambda(report_stage))
        state = chain.invoke({"tickets": tickets})
        orchestrator = "langchain"
    except Exception as exc:
        state = {"tickets": tickets}
        for stage in (classify_stage, enrich_stage, summarize_stage, report_stage):
            state = stage(state)
        if not isinstance(exc, ImportError):
            print(f"[agent] LangChain error ({exc}); ran plain-sequence fallback.", file=sys.stderr)

    print(f"[agent] classifier mode: {clf.mode}  |  orchestration: {orchestrator}  |  "
          f"tickets: {len(tickets)}", file=sys.stderr)
    return {"report": state["report"], "report_path": state.get("report_path"),
            "processed": state["processed"], "summary": state["summary"], "mode": clf.mode}


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
