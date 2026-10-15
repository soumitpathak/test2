import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA = Path("output/synthetic_dataset.csv")
REPORT = Path("output/validation_report.json")
MIN_CASES = 100

REQUIRED = [
    "case_id", "domain", "intent", "input", "expected_behavior",
    "reference_context", "test_type", "risk_level", "source", "seed_id",
    "review_status", "generation_model", "dataset_version", "created_at"
]
TEST_TYPES = {
    "paraphrase", "ambiguous", "incomplete", "edge_case", "negative",
    "adversarial", "typo", "context_conflict", "multi_intent", "boundary_condition"
}
RISKS = {"low", "medium", "high", "critical"}


def normalize(value):
    value = str(value or "").lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)
    return value


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found. Run python generate_dataset.py first.")

    df = pd.read_csv(DATA).fillna("")
    errors, warnings = [], []

    missing_columns = sorted(set(REQUIRED) - set(df.columns))
    if missing_columns:
        errors.append({"gate": "schema", "message": f"Missing columns: {missing_columns}"})
    else:
        required_nonblank = ["case_id", "domain", "intent", "input", "expected_behavior", "test_type", "risk_level"]
        for col in required_nonblank:
            count = int((df[col].astype(str).str.strip() == "").sum())
            if count:
                errors.append({"gate": "required_values", "message": f"{count} blank values in {col}"})

        duplicate_ids = int(df["case_id"].duplicated().sum())
        if duplicate_ids:
            errors.append({"gate": "unique_ids", "message": f"{duplicate_ids} duplicate case_id values"})

        normalized = df["input"].map(normalize)
        duplicate_inputs = int(normalized.duplicated().sum())
        if duplicate_inputs:
            errors.append({"gate": "unique_inputs", "message": f"{duplicate_inputs} duplicate normalized inputs remain"})

        invalid_types = sorted(set(df["test_type"].astype(str)) - TEST_TYPES)
        if invalid_types:
            errors.append({"gate": "test_type", "message": f"Unsupported test_type values: {invalid_types}"})

        invalid_risks = sorted(set(df["risk_level"].astype(str)) - RISKS)
        if invalid_risks:
            errors.append({"gate": "risk_level", "message": f"Unsupported risk_level values: {invalid_risks}"})

    if len(df) < MIN_CASES:
        errors.append({"gate": "minimum_volume", "message": f"Expected at least {MIN_CASES} cases; found {len(df)}"})

    observed_types = set(df["test_type"].astype(str)) if "test_type" in df.columns else set()
    missing_types = sorted(TEST_TYPES - observed_types)
    if missing_types:
        warnings.append({"gate": "test_type_coverage", "message": f"No cases found for: {missing_types}"})

    report = {
        "dataset_path": str(DATA),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": int(len(df)),
        "unique_case_ids": int(df["case_id"].nunique()) if "case_id" in df.columns else 0,
        "test_types_present": sorted(observed_types),
        "errors": errors,
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL",
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()