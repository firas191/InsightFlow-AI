"""Agentic extension: tools the assistant can call.

The brief asks for tool-calling that connects natural language to business
systems. These three tools are the bridge:

  * knowledge_search       — semantic lookup over resolved historical cases
  * create_followup_ticket — (mock) files a follow-up task into a local queue
  * export_report          — writes the daily pulse report to disk

Each is a plain Python function (easy to unit-test and to call from the
deterministic pipeline) AND is exposed as a LangChain Tool via
`build_langchain_tools()` so an LLM agent can call them by name. LangChain is
optional: if it isn't installed, the plain functions still work.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.config import CFG
from src.retriever import CaseRetriever

_RETRIEVER: CaseRetriever | None = None


def _retriever() -> CaseRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = CaseRetriever()
    return _RETRIEVER


# --- Tool 1: knowledge search ----------------------------------------------
def knowledge_search(query: str, k: int | None = None) -> list[dict]:
    """Find the most similar resolved historical cases for a ticket/query."""
    return _retriever().retrieve(query, k=k)


# --- Tool 2: create a follow-up ticket (mock business system) ----------------
def create_followup_ticket(summary: str, team: str, urgency: str = "Medium") -> dict:
    """File a follow-up ticket into a local queue (stands in for a real ticketing API)."""
    reports_dir = CFG.path("paths", "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    queue = reports_dir / "followup_queue.jsonl"
    ref = f"FUP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    record = {
        "ref": ref,
        "summary": summary,
        "team": team,
        "urgency": urgency,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "queued",
    }
    with open(queue, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


# --- Tool 3: export the daily report ----------------------------------------
def export_report(markdown: str, filename: str | None = None) -> str:
    """Write the daily pulse report to the reports directory; returns the path."""
    reports_dir = CFG.path("paths", "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    filename = filename or f"pulse_{datetime.now().strftime('%Y-%m-%d')}.md"
    path = reports_dir / filename
    path.write_text(markdown, encoding="utf-8")
    return str(path)


# --- Optional LangChain wrappers --------------------------------------------
def build_langchain_tools():
    """Return these capabilities as LangChain Tools (for use in an LLM agent).

    Returns an empty list (with a printed note) if LangChain isn't installed,
    so importing this module never hard-fails.
    """
    try:
        from langchain.tools import StructuredTool
    except Exception:  # pragma: no cover - optional dep
        print("[tools] LangChain not installed — plain Python tools still available.")
        return []

    return [
        StructuredTool.from_function(
            func=knowledge_search,
            name="knowledge_search",
            description="Search resolved historical support cases for ones similar to a query. "
                        "Input: a ticket description string. Returns ranked cases with resolutions.",
        ),
        StructuredTool.from_function(
            func=create_followup_ticket,
            name="create_followup_ticket",
            description="File a follow-up ticket for a support team. "
                        "Inputs: summary (str), team (str), urgency (str).",
        ),
        StructuredTool.from_function(
            func=export_report,
            name="export_report",
            description="Persist the daily pulse report. Inputs: markdown (str), filename (optional str). "
                        "Returns the saved file path.",
        ),
    ]
