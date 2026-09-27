#!/usr/bin/env python3
"""
magicpin AI Challenge — Generate Submission
============================================
Generates the canonical 30-line submission.jsonl from dataset/expanded/test_pairs.json
using the grounded 4-context composition engine in bot.py.

Usage:
    python generate_submission.py
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# Ensure root directory is on path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

import bot


def load_json(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    expanded_dir = BASE_DIR / "dataset" / "expanded"
    if not expanded_dir.exists():
        print(f"Error: {expanded_dir} does not exist. Run python dataset/generate_dataset.py first.")
        sys.exit(1)

    test_pairs_path = expanded_dir / "test_pairs.json"
    if not test_pairs_path.exists():
        print(f"Error: {test_pairs_path} not found.")
        sys.exit(1)

    test_pairs_data = load_json(test_pairs_path)
    pairs = test_pairs_data.get("pairs", [])
    print(f"Loaded {len(pairs)} test pairs from {test_pairs_path}")

    # Load categories
    categories = {}
    cat_dir = expanded_dir / "categories"
    for cat_file in cat_dir.glob("*.json"):
        cat_data = load_json(cat_file)
        categories[cat_data.get("slug")] = cat_data

    # Load merchants
    merchants = {}
    m_dir = expanded_dir / "merchants"
    for m_file in m_dir.glob("*.json"):
        m_data = load_json(m_file)
        merchants[m_data.get("merchant_id")] = m_data

    # Load triggers
    triggers = {}
    trg_dir = expanded_dir / "triggers"
    for trg_file in trg_dir.glob("*.json"):
        trg_data = load_json(trg_file)
        triggers[trg_data.get("id")] = trg_data

    # Load customers
    customers = {}
    c_dir = expanded_dir / "customers"
    for c_file in c_dir.glob("*.json"):
        c_data = load_json(c_file)
        customers[c_data.get("customer_id")] = c_data

    submission_lines = []

    for pair in pairs:
        t_id = pair["test_id"]
        trg_id = pair["trigger_id"]
        m_id = pair["merchant_id"]
        c_id = pair.get("customer_id")

        trg = triggers.get(trg_id)
        if not trg:
            print(f"Warning: Trigger {trg_id} not found for {t_id}")
            continue

        merchant = merchants.get(m_id)
        if not merchant:
            print(f"Warning: Merchant {m_id} not found for {t_id}")
            continue

        cat_slug = merchant.get("category_slug", "")
        category = categories.get(cat_slug)

        customer = customers.get(c_id) if c_id else None

        composed = bot.compose(
            category=category,
            merchant=merchant,
            trigger=trg,
            customer=customer
        )

        if not composed:
            # Preserve the canonical pair while explicitly recording a safe omission.
            # A blank body plus cta=none means no message should be sent for this case.
            submission_lines.append({
                "test_id": t_id,
                "body": "",
                "cta": "none",
                "send_as": "vera",
                "suppression_key": trg.get("suppression_key") or f"omit:{trg_id}",
                "rationale": "No message emitted because required verified facts, consent, or a supported trigger pattern are unavailable."
            })
            print(f"Omitted {t_id}: required context or consent is missing")
            continue

        submission_lines.append({
            "test_id": t_id,
            "body": composed["body"],
            "cta": composed["cta"],
            "send_as": composed["send_as"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"]
        })

    out_file = BASE_DIR / "submission.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for item in submission_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(submission_lines)} lines in {out_file}")


if __name__ == "__main__":
    main()
