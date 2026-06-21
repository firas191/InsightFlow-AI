"""Click-through demo for the Enterprise Pulse Agent.

Run from the repo root:
    streamlit run app/streamlit_app.py

Lets you classify a single ticket or run the full pipeline over a CSV and view
the generated daily pulse report. Designed for a presentation demo.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the repo importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CFG  # noqa: E402

st.set_page_config(page_title="InsightFlow — Enterprise Pulse Agent", layout="wide")
st.title("InsightFlow — Enterprise Pulse Agent")
st.caption("Classify → retrieve → summarize → draft → daily pulse. "
           "Outputs are suggestions and require human review.")


@st.cache_resource(show_spinner="Loading classifier…")
def get_classifier(prefer_finetuned: bool):
    from src.classifier.inference import TicketClassifier
    return TicketClassifier(prefer_finetuned=prefer_finetuned)


tab_single, tab_batch = st.tabs(["Single ticket", "Batch → daily pulse"])

with tab_single:
    prefer = not st.checkbox("Force zero-shot fallback", value=False)
    text = st.text_area("Ticket text", height=140,
                        value="I was charged twice for my subscription and need this fixed ASAP.")
    if st.button("Classify", type="primary"):
        clf = get_classifier(prefer)
        st.info(f"Classifier mode: **{clf.mode}**")
        pred = clf.classify([text])[0]
        category = pred["category"]["label"]
        team = CFG.get("category_to_team", default={}).get(category, "Customer Success")
        cols = st.columns(4)
        for col, task in zip(cols[:3], ["category", "intent", "sentiment"]):
            col.metric(task.title(), pred[task]["label"], f"{pred[task]['confidence']:.0%}")
        cols[3].metric("Routed team", team, "rule-based")

        from src.retriever import CaseRetriever
        from src.responder import ResponseDrafter
        cases = CaseRetriever().retrieve(text)
        st.subheader("Similar resolved cases")
        st.dataframe(pd.DataFrame(cases)[["case_id", "category", "problem", "score"]],
                     use_container_width=True)
        st.subheader("Suggested draft response")
        st.code(ResponseDrafter().draft(text, pred, cases))

with tab_batch:
    default_csv = CFG.get("paths", "sample_csv", default="data/sample_tickets.csv")
    uploaded = st.file_uploader("Upload a ticket CSV (needs a 'text' column)", type="csv")
    limit = st.slider("Max tickets to process", 5, 100, 30)
    if st.button("Run daily pulse", type="primary"):
        from src.pulse_agent import load_tickets, run_pipeline
        if uploaded is not None:
            tmp = ROOT / "data" / "_uploaded.csv"
            tmp.write_bytes(uploaded.getvalue())
            tickets = load_tickets(tmp, limit)
        else:
            tickets = load_tickets(ROOT / default_csv, limit)
        with st.spinner("Running pipeline…"):
            result = run_pipeline(tickets, export=True)
        st.success(f"Done — classifier mode: {result['mode']}. "
                   f"Report saved to {result['report_path']}")
        st.markdown(result["report"])
