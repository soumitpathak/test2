import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT = Path("output")
DATA = OUT / "synthetic_dataset.csv"
VALIDATION = OUT / "validation_report.json"
REPO = Path("datasets")

if not DATA.exists() or not VALIDATION.exists():
    raise FileNotFoundError("Run generation and validation before promotion.")

validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
if validation.get("status") != "PASS":
    raise SystemExit("Promotion blocked: validation status is not PASS.")

df = pd.read_csv(DATA).fillna("")
if df.empty:
    raise SystemExit("Promotion blocked: dataset is empty.")

versions = df["dataset_version"].astype(str).unique().tolist()
if len(versions) != 1:
    raise SystemExit(f"Promotion blocked: expected one dataset version, found {versions}")
version = versions[0]

target = REPO / "customer_support" / f"v{version}"
if target.exists():
    raise SystemExit(f"Promotion blocked: {target} already exists. Use a new version; do not overwrite.")
target.mkdir(parents=True)

shutil.copy2(DATA, target / "evaluation_dataset.csv")
shutil.copy2(OUT / "synthetic_dataset.jsonl", target / "evaluation_dataset.jsonl")

manifest = {
    "dataset_name": "customer_support_evaluation",
    "dataset_version": version,
    "domain": "customer_support",
    "record_count": int(len(df)),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "generation_models": sorted(df["generation_model"].astype(str).unique().tolist()),
    "validation_status": validation["status"],
    "validation_report": str(VALIDATION),
    "approval_status": "pending_human_review",
    "note": "LLM-generated cases require human review before use as a trusted gold evaluation set."
}
(target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Dataset staged at: {target}")
print("Status: pending human review; staging is not equivalent to production approval.")