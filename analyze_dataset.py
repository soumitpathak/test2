import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA = Path("output/synthetic_dataset.csv")
OUT = Path("output")
df = pd.read_csv(DATA).fillna("")


def counts(column):
    return {str(k): int(v) for k, v in df[column].value_counts(dropna=False).items()}


report = {
    "analyzed_at": datetime.now(timezone.utc).isoformat(),
    "dataset_version": str(df["dataset_version"].iloc[0]) if len(df) else None,
    "record_count": int(len(df)),
    "unique_inputs": int(df["input"].astype(str).str.lower().str.strip().nunique()),
    "by_domain": counts("domain"),
    "by_intent": counts("intent"),
    "by_test_type": counts("test_type"),
    "by_risk_level": counts("risk_level"),
    "by_review_status": counts("review_status"),
    "high_or_critical_cases": int(df["risk_level"].isin(["high", "critical"]).sum()),
    "pending_review_cases": int((df["review_status"] == "pending").sum()),
}
(OUT / "coverage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

for column in ["test_type", "intent", "risk_level", "review_status"]:
    df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="count").to_csv(
        OUT / f"coverage_{column}.csv", index=False
    )

print(json.dumps(report, indent=2))
print("\nReview the CSV reports. A balanced dataset is not automatically a correct dataset.")