"""Generate a synthetic, labeled support-ticket dataset for InsightFlow AI.

Produces tickets with four labels (category, urgency, team, sentiment) using
templated, category-conditioned text so a transformer can learn real signal.

The label schema mirrors config/config.yaml (the canonical source). This script
embeds an identical copy so it runs standalone — including inside the Kaggle
training notebook, where the repo may not be importable.

Usage:
    python data/generate_synthetic_data.py --n 4000 --out data/
    python data/generate_synthetic_data.py --n 200 --out data/ --sample-only
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

# --- Canonical label schema (mirrors config/config.yaml) -------------------
CATEGORIES = [
    "Billing", "Technical Issue", "Account Access", "Feature Request",
    "Shipping & Delivery", "Product Quality", "General Inquiry",
]
URGENCIES = ["Low", "Medium", "High", "Critical"]
TEAMS = [
    "Billing Team", "Technical Support", "Account Security",
    "Product Team", "Logistics Team", "Customer Success",
]
SENTIMENTS = ["Negative", "Neutral", "Positive"]

CATEGORY_TO_TEAM = {
    "Billing": "Billing Team",
    "Technical Issue": "Technical Support",
    "Account Access": "Account Security",
    "Feature Request": "Product Team",
    "Shipping & Delivery": "Logistics Team",
    "Product Quality": "Product Team",
    "General Inquiry": "Customer Success",
}

PRODUCTS = [
    "the mobile app", "the web dashboard", "the API", "the billing portal",
    "the analytics module", "the export tool", "the integration", "my account",
    "the desktop client", "the reporting feature", "the checkout flow", "the sync service",
]

# Per-category problem phrasings.
CATEGORY_PHRASES = {
    "Billing": [
        "I was charged twice for my subscription this month",
        "my invoice shows an amount I don't recognize",
        "the refund I was promised hasn't arrived",
        "my payment keeps getting declined even though the card is valid",
        "I want to cancel my plan but I'm still being billed",
        "the pricing on my latest invoice doesn't match what I signed up for",
        "I need a copy of last quarter's invoices for accounting",
    ],
    "Technical Issue": [
        "I keep getting a 500 error when I try to load {product}",
        "{product} crashes every time I upload a file",
        "the API returns a timeout under moderate load",
        "data isn't syncing between {product} and our CRM",
        "I'm seeing a blank screen after the latest update",
        "webhooks stopped firing this morning",
        "the integration with Slack broke after we rotated our keys",
    ],
    "Account Access": [
        "I can't log in even after resetting my password",
        "I'm locked out of my account after too many attempts",
        "two-factor authentication isn't sending me a code",
        "SSO redirects me in a loop and never signs me in",
        "I never received the password reset email",
        "my account says it's suspended but I don't know why",
        "a teammate lost access after we changed the admin owner",
    ],
    "Feature Request": [
        "it would be great if {product} supported dark mode",
        "please consider adding bulk export to CSV",
        "we'd love an option to schedule reports automatically",
        "could you add role-based permissions for larger teams",
        "a mobile widget for quick stats would be really helpful",
        "it would help if we could tag tickets with custom labels",
        "an integration with Microsoft Teams is on our wishlist",
    ],
    "Shipping & Delivery": [
        "my order has been stuck in transit for over a week",
        "the tracking number doesn't update at all",
        "I received the wrong item in my package",
        "my delivery was marked delivered but nothing arrived",
        "the estimated delivery date keeps slipping",
        "part of my order is missing from the box",
        "I need to change the shipping address before it ships",
    ],
    "Product Quality": [
        "the unit arrived damaged and won't power on",
        "the product stopped working after only two days",
        "there are missing parts in the package I received",
        "the build quality feels far below what was advertised",
        "the screen has a visible defect out of the box",
        "the item doesn't match the description on the website",
        "the device overheats during normal use",
    ],
    "General Inquiry": [
        "I just want to understand how {product} handles data retention",
        "what are your support hours in the EU timezone",
        "can you point me to documentation for the export format",
        "I'm evaluating your product and have a few questions",
        "how do I add another seat to our team plan",
        "is there a way to contact sales about volume pricing",
        "could you clarify the difference between the Pro and Business tiers",
    ],
}

# Tone modifiers by sentiment.
SENTIMENT_OPENERS = {
    "Negative": [
        "I'm really frustrated.", "This is unacceptable.", "I'm very disappointed.",
        "I've had enough of this.", "Honestly this is a mess.",
    ],
    "Neutral": [
        "Hi,", "Hello,", "Quick question —", "Reaching out because", "FYI,",
    ],
    "Positive": [
        "Thanks for the great product!", "Love the tool overall —",
        "Appreciate your help in advance.", "You folks are usually great, but",
        "Big fan here, just one thing:",
    ],
}

# Urgency cues appended to raise/lower perceived priority.
URGENCY_TAILS = {
    "Critical": [
        "Our entire team is blocked and we're losing revenue every hour.",
        "This is a production outage affecting all our customers.",
        "We may have a data-loss situation — please escalate immediately.",
        "This is business-critical and needs attention right now.",
    ],
    "High": [
        "We have a launch deadline tomorrow, so this is urgent.",
        "Please prioritize this — it's blocking my work.",
        "I need this resolved as soon as possible.",
        "This is becoming a serious problem for us.",
    ],
    "Medium": [
        "Hoping to get this sorted out soon.",
        "It's not blocking everything, but it's slowing us down.",
        "Would appreciate a fix this week.",
        "Let me know what the next steps are.",
    ],
    "Low": [
        "No rush on this one.",
        "Whenever you get a chance is fine.",
        "Just flagging it for the record.",
        "Minor thing, low priority.",
    ],
}

# Category-conditioned distributions make the heads correlated but non-trivial.
# weights map to [Low, Medium, High, Critical] and [Negative, Neutral, Positive].
URGENCY_WEIGHTS = {
    "Billing":            [0.15, 0.40, 0.30, 0.15],
    "Technical Issue":    [0.05, 0.25, 0.40, 0.30],
    "Account Access":     [0.05, 0.30, 0.40, 0.25],
    "Feature Request":    [0.55, 0.35, 0.08, 0.02],
    "Shipping & Delivery":[0.15, 0.45, 0.30, 0.10],
    "Product Quality":    [0.10, 0.35, 0.35, 0.20],
    "General Inquiry":    [0.55, 0.35, 0.08, 0.02],
}
SENTIMENT_WEIGHTS = {
    "Billing":            [0.55, 0.35, 0.10],
    "Technical Issue":    [0.60, 0.30, 0.10],
    "Account Access":     [0.55, 0.35, 0.10],
    "Feature Request":    [0.10, 0.45, 0.45],
    "Shipping & Delivery":[0.55, 0.35, 0.10],
    "Product Quality":    [0.70, 0.22, 0.08],
    "General Inquiry":    [0.10, 0.70, 0.20],
}

# Probability that a ticket is "mis-routed" to a non-default team (realistic noise).
TEAM_NOISE = 0.08


def _pick(rng: random.Random, items, weights=None):
    if weights:
        return rng.choices(items, weights=weights, k=1)[0]
    return rng.choice(items)


def make_ticket(rng: random.Random) -> dict:
    category = _pick(rng, CATEGORIES)
    urgency = _pick(rng, URGENCIES, URGENCY_WEIGHTS[category])
    sentiment = _pick(rng, SENTIMENTS, SENTIMENT_WEIGHTS[category])

    # Team: usually the default mapping, occasionally noisy.
    if rng.random() < TEAM_NOISE:
        team = _pick(rng, TEAMS)
    else:
        team = CATEGORY_TO_TEAM[category]

    product = _pick(rng, PRODUCTS)
    body = _pick(rng, CATEGORY_PHRASES[category]).format(product=product)
    opener = _pick(rng, SENTIMENT_OPENERS[sentiment])
    tail = _pick(rng, URGENCY_TAILS[urgency])

    text = f"{opener} {body}. {tail}".strip()
    # Light cleanup of doubled punctuation.
    text = text.replace("..", ".").replace(" .", ".")

    return {
        "text": text,
        "category": category,
        "urgency": urgency,
        "team": team,
        "sentiment": sentiment,
    }


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = [make_ticket(rng) for _ in range(n)]
    df = pd.DataFrame(rows)
    # Drop exact-duplicate texts to keep splits clean.
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df.insert(0, "ticket_id", [f"TK-{i:05d}" for i in range(len(df))])
    return df


def split(df: pd.DataFrame, seed: int = 42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    return (
        df.iloc[:n_train].reset_index(drop=True),
        df.iloc[n_train:n_train + n_val].reset_index(drop=True),
        df.iloc[n_train + n_val:].reset_index(drop=True),
    )


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic support tickets.")
    ap.add_argument("--n", type=int, default=4000, help="number of tickets to generate")
    ap.add_argument("--out", type=str, default="data", help="output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-only", action="store_true",
                    help="write a single sample_tickets.csv instead of train/val/test")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = generate(args.n, args.seed)

    if args.sample_only:
        df.to_csv(out / "sample_tickets.csv", index=False)
        print(f"Wrote {len(df)} rows -> {out / 'sample_tickets.csv'}")
        return

    train, val, test = split(df, args.seed)
    train.to_csv(out / "tickets_train.csv", index=False)
    val.to_csv(out / "tickets_val.csv", index=False)
    test.to_csv(out / "tickets_test.csv", index=False)
    print(f"Wrote splits to {out}/  "
          f"train={len(train)} val={len(val)} test={len(test)}")
    # Quick label balance report.
    for col in ["category", "urgency", "team", "sentiment"]:
        print(f"\n[{col}] distribution:")
        print(df[col].value_counts().to_string())


if __name__ == "__main__":
    main()
