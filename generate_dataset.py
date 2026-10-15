import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
VERSION = os.getenv("DATASET_VERSION", "1.0.0")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

client = OpenAI(api_key=API_KEY)
OUT = Path("output")
OUT.mkdir(exist_ok=True)

TEST_TYPES = [
    "paraphrase", "ambiguous", "incomplete", "edge_case", "negative",
    "adversarial", "typo", "context_conflict", "multi_intent", "boundary_condition"
]
ALLOWED_RISK = {"low", "medium", "high", "critical"}


def normalize(value):
    value = str(value or "").lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)
    return value


def generate_for_seed(seed):
    prompt = f"""
You are creating synthetic evaluation cases for an enterprise customer-support AI assistant.
Use the seed scenario as grounding, but create realistic and meaningfully different user inputs.
Return ONLY a valid JSON object with key "cases" containing exactly 10 objects.

Seed:
{json.dumps(seed, ensure_ascii=False)}

Each object must contain:
- input: a realistic user message; do not include real personal data or secrets
- intent: concise intent label
- expected_behavior: testable behavioral requirement, not a generic answer
- test_type: exactly one of {TEST_TYPES}
- risk_level: one of low, medium, high, critical
- reference_context: concise policy/context needed to judge the response

Requirements:
1. Include broad language variation and difficult cases, not just simple paraphrases.
2. At least one case should require clarification; at least one should test privacy or security when relevant.
3. Adversarial cases must test instruction manipulation without providing harmful operational instructions.
4. Do not invent company-specific policies beyond the supplied seed context.
5. Expected behavior must be specific enough for a human evaluator to judge pass/fail.
6. Keep each input under 500 characters and each expected_behavior under 500 characters.
7. Use all 10 test_type values once each across the 10 generated objects.
"""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You generate high-quality synthetic AI evaluation datasets. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    payload = json.loads(response.choices[0].message.content)
    cases = payload.get("cases", [])
    if len(cases) != 10:
        raise ValueError(f"{seed['seed_id']}: expected 10 cases, received {len(cases)}")

    now = datetime.now(timezone.utc).isoformat()
    enriched = []
    for idx, case in enumerate(cases, start=1):
        test_type = str(case.get("test_type", "")).strip().lower()
        risk = str(case.get("risk_level", seed["risk_level"])).strip().lower()
        if test_type not in TEST_TYPES:
            test_type = TEST_TYPES[(idx - 1) % len(TEST_TYPES)]
        if risk not in ALLOWED_RISK:
            risk = seed["risk_level"] if seed["risk_level"] in ALLOWED_RISK else "medium"

        enriched.append({
            "case_id": f"{seed['seed_id']}-{idx:02d}",
            "domain": seed["domain"],
            "intent": str(case.get("intent") or seed["intent"]).strip(),
            "input": str(case.get("input", "")).strip(),
            "expected_behavior": str(case.get("expected_behavior", "")).strip(),
            "reference_context": str(case.get("reference_context") or seed["reference_context"]).strip(),
            "test_type": test_type,
            "risk_level": risk,
            "source": "llm_synthetic",
            "seed_id": seed["seed_id"],
            "review_status": "pending",
            "generation_model": MODEL,
            "dataset_version": VERSION,
            "created_at": now,
        })
    return enriched


def main():
    seeds = pd.read_csv("seed_cases.csv").fillna("").to_dict(orient="records")
    required_seed_cols = {
        "seed_id", "domain", "intent", "seed_input", "reference_context",
        "expected_behavior", "risk_level"
    }
    if not seeds or not required_seed_cols.issubset(seeds[0].keys()):
        raise ValueError(f"seed_cases.csv must contain columns: {sorted(required_seed_cols)}")

    all_cases = []
    for index, seed in enumerate(seeds, start=1):
        print(f"[{index}/{len(seeds)}] Generating from {seed['seed_id']} ({seed['intent']})")
        try:
            all_cases.extend(generate_for_seed(seed))
        except Exception as exc:
            print(f"ERROR for {seed['seed_id']}: {exc}")
            raise
        time.sleep(0.3)  # gentle pacing; production pipelines should implement retry/backoff

    raw = pd.DataFrame(all_cases)
    raw.to_csv(OUT / "synthetic_dataset_raw.csv", index=False)
    raw.to_json(OUT / "synthetic_dataset_raw.jsonl", orient="records", lines=True, force_ascii=False)

    raw["normalized_input"] = raw["input"].map(normalize)
    raw["is_duplicate"] = raw.duplicated("normalized_input", keep="first")
    duplicates = raw[raw["is_duplicate"]].copy()
    clean = raw[~raw["is_duplicate"]].drop(columns=["normalized_input", "is_duplicate"]).copy()

    duplicates.to_csv(OUT / "duplicate_candidates.csv", index=False)
    clean.to_csv(OUT / "synthetic_dataset.csv", index=False)
    clean.to_json(OUT / "synthetic_dataset.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"Raw cases: {len(raw)}")
    print(f"Unique cases: {len(clean)}")
    print(f"Duplicate candidates: {len(duplicates)}")
    print(f"Dataset version: {VERSION}")
    print("Next: python validate_dataset.py")


if __name__ == "__main__":
    main()