"""Draft suggested responses for support agents.

Template-driven (not free-form generation) for safety: the brief requires that
nothing unsupported is asserted and nothing is auto-sent. Drafts are grounded in
(a) the predicted category and (b) the resolution text of the most similar
historical case. Every output is explicitly labeled a DRAFT for human review.
"""
from __future__ import annotations

from src.config import CFG

_GREETING = "Hi there,"
_CLOSING = "Best regards,\nInsightFlow Support"

# Keyed by Bitext category. Falls back to a generic opener.
_OPENERS = {
    "ACCOUNT": "Thanks for reaching out about your account — let's get this sorted safely.",
    "CANCEL": "Thanks for reaching out about cancelling — let's take care of it.",
    "CONTACT": "Thanks for getting in touch — I'll make sure this reaches the right person.",
    "DELIVERY": "Thanks for letting us know about your delivery — let's track this down.",
    "FEEDBACK": "Thanks for taking the time to share this feedback — it genuinely helps.",
    "INVOICE": "Thanks for your invoice question — I can help with that.",
    "ORDER": "Thanks for contacting us about your order — let's take care of it.",
    "PAYMENT": "Thanks for flagging this payment issue — I understand how important it is.",
    "REFUND": "Thanks for reaching out about your refund — let's get it moving.",
    "SHIPPING": "Thanks for the note about your shipping — let's sort it out.",
    "SUBSCRIPTION": "Thanks for reaching out about your subscription — happy to help.",
}
_GENERIC_OPENER = "Thanks for getting in touch — happy to help."


class ResponseDrafter:
    def draft(self, ticket_text: str, prediction: dict, similar_cases: list[dict]) -> str:
        category = prediction.get("category", {}).get("label", "")
        sentiment = prediction.get("sentiment", {}).get("label", "Neutral")
        opener = _OPENERS.get(category, _GENERIC_OPENER)

        if similar_cases:
            top = similar_cases[0]
            grounded = (
                f"Based on a similar past case ({top['case_id']}), the likely fix is: "
                f"{top['resolution']}"
            )
        else:
            grounded = "I'm escalating this to the right team and will follow up shortly."

        priority = set(CFG.get("agent", "priority_categories", default=[]))
        priority_line = ""
        if sentiment == "Negative" or category in priority:
            priority_line = "\n\nI've flagged this as a priority given the impact described."

        body = f"{opener} {grounded}{priority_line}"
        return f"[DRAFT — review before sending]\n\n{_GREETING}\n\n{body}\n\n{_CLOSING}"
