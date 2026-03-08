#!/usr/bin/env python3
"""
Layer 1A (Offline-1): Preserve columns + generate table descriptions (LLM)

INPUT:
- tables_raw/T1.json ... tables_raw/T10.json
  (These are created by Layer 0.)

OUTPUT:
1) tables_clean/Ti.json  (one file per table)
   Each output file contains:
   - table_id, source_table_id, domain
   - title
   - columns (preserved)
   - rows (kept as-is)
   - description + numeric/categorical/entities predicted by the LLM

2) schema_descriptions.json (one combined summary file for all 10 tables)
"""

import os
import json
from typing import Any, Dict, List, Optional

from litellm import completion


# ----------------------------
# Config
# ----------------------------

API_BASE = "https://thekeymaker.umass.edu/"
MODEL_NAME = "openai/gpt4o"

TABLES_RAW_DIR = "tables_raw"
TABLES_CLEAN_DIR = "../tables_clean"
SCHEMA_DESCRIPTIONS_PATH = "schema_descriptions.json"

# We show only a few rows to the LLM so prompts stay small and cheap.
NUM_SAMPLE_ROWS = 8

# If the model returns something not valid JSON, we retry a couple times.
MAX_RETRIES = 2


# ----------------------------
# Helper functions
# ----------------------------

def load_json(path: str) -> Dict[str, Any]:
    """Read a JSON file and return it."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    """Write an object to a JSON file in a readable format."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def preview_rows(rows: List[List[Any]], max_rows: int) -> List[List[Any]]:
    """Return only the first few rows so we don't send huge tables to the LLM."""
    return rows[:max_rows] if rows else []


def build_llm_messages(
    table_id: str,
    title: str,
    domain: str,
    columns: List[str],
    sample_rows: List[List[Any]],
    num_rows: int,
    num_cols: int
) -> List[Dict[str, str]]:
    """
    Build a prompt that tells the LLM to return JSON.
    We also tell it to use column names exactly as given.
    """

    # System message: sets strict rules about output format
    system_msg = (
        "Return ONLY valid JSON. "
        "No extra text. No markdown. No backticks. "
        "Use the provided column names exactly (do not rename columns)."
    )

    # We send the LLM a compact, structured view of the table
    user_payload = {
        "table_id": table_id,
        "title": title,
        "domain": domain,
        "num_rows": num_rows,
        "num_columns": num_cols,
        "columns_original": columns,
        "sample_rows": sample_rows,
        "task": (
            "Generate a short description of the table and classify columns. "
            "Use ONLY the column names from columns_original."
        ),
        "return_json_keys": [
            "description",
            "numeric_columns",
            "categorical_columns",
            "entities"
        ],
        "rules": [
            "description must be 2-4 sentences and grounded in the table.",
            "numeric_columns: column names that mostly contain numbers (counts, scores, money, etc.).",
            "categorical_columns: column names that represent categories/labels/text groups.",
            "entities: column names that represent real-world things (people, teams, locations, organizations, items).",
            "Only use column names from columns_original in the lists.",
            "If unsure, return empty lists."
        ],
        "example_output": {
            "description": "This table ...",
            "numeric_columns": ["Revenue", "Year"],
            "categorical_columns": ["Region"],
            "entities": ["Region"]
        }
    }

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]


def normalize_list_of_columns(cols: Any) -> List[str]:
    """
    Make sure we always return a list of strings.
    If the model returns something weird, we fall back to [].
    """
    if not isinstance(cols, list):
        return []
    out: List[str] = []
    for x in cols:
        if isinstance(x, str):
            out.append(x)
    return out


def call_llm(api_key: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Call the LLM and parse JSON.
    Retries if the response isn't valid JSON.
    """
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = completion(
                model=MODEL_NAME,
                messages=messages,
                api_base=API_BASE,
                api_key=api_key,
                temperature=0.2
            )

            text = resp["choices"][0]["message"]["content"].strip()
            parsed = json.loads(text)

            # Check required keys exist
            required = ["description", "numeric_columns", "categorical_columns", "entities"]
            for k in required:
                if k not in parsed:
                    raise ValueError(f"Missing key in LLM output: {k}")

            # Clean types
            if not isinstance(parsed["description"], str):
                parsed["description"] = str(parsed["description"])

            parsed["numeric_columns"] = normalize_list_of_columns(parsed["numeric_columns"])
            parsed["categorical_columns"] = normalize_list_of_columns(parsed["categorical_columns"])
            parsed["entities"] = normalize_list_of_columns(parsed["entities"])

            return parsed

        except Exception as e:
            last_error = e

            # If we can retry, add a correction message
            if attempt < MAX_RETRIES:
                messages = messages + [{
                    "role": "user",
                    "content": (
                        "Your last response was not valid JSON or missed required keys. "
                        "Return ONLY valid JSON with keys: description, numeric_columns, categorical_columns, entities."
                    )
                }]
            else:
                break

    raise RuntimeError(f"LLM failed after retries. Last error: {last_error}")


# ----------------------------
# Main function
# ----------------------------

def main() -> None:
    """
    For T1..T10:
    - Read raw table JSON
    - Preserve columns + domain
    - Call LLM to get description + column groups
    - Save per-table clean JSON
    - Save one combined schema_descriptions.json
    """
    api_key = os.environ.get("LITELLM_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing LITELLM_API_KEY environment variable.\n"
            "Set it like:\n"
            "  export LITELLM_API_KEY='YOUR_KEY'\n"
            "Or PowerShell:\n"
            "  $env:LITELLM_API_KEY='YOUR_KEY'"
        )

    os.makedirs(TABLES_CLEAN_DIR, exist_ok=True)

    # Combined output summary for all tables
    all_schema: Dict[str, Any] = {}

    for i in range(1, 11):
        table_id = f"T{i}"
        raw_path = os.path.join(TABLES_RAW_DIR, f"{table_id}.json")

        if not os.path.exists(raw_path):
            print(f"[WARN] Missing file: {raw_path} (skipping)")
            continue

        raw_obj = load_json(raw_path)

        # Layer 0 raw format: domain + table blob
        domain = raw_obj.get("domain", "Unknown")
        table_blob = raw_obj.get("table", {})

        title = table_blob.get("title") or ""
        columns_original = table_blob.get("header") or []
        rows = table_blob.get("data") or []

        num_rows = len(rows)
        num_cols = len(columns_original)

        # Send only a few example rows to the LLM
        sample_rows = preview_rows(rows, NUM_SAMPLE_ROWS)

        # Build prompt and call the LLM
        messages = build_llm_messages(
            table_id=table_id,
            title=title,
            domain=domain,
            columns=columns_original,
            sample_rows=sample_rows,
            num_rows=num_rows,
            num_cols=num_cols
        )

        print(f"[INFO] {table_id}: calling LLM (rows={num_rows}, cols={num_cols}, domain={domain})...")
        llm_out = call_llm(api_key=api_key, messages=messages)

        # Build the clean table object (preserving columns + domain)
        clean_obj = {
            "table_id": table_id,
            "source": raw_obj.get("source", "HybridQA"),
            "source_table_id": raw_obj.get("source_table_id"),
            "domain": domain,
            "title": title,

            # Preserve columns exactly (do not rename)
            "columns": columns_original,

            # Keep all rows (can be useful later for SQL / grounding)
            "rows": rows,

            # LLM-generated schema info
            "description": llm_out["description"],
            "numeric_columns": llm_out["numeric_columns"],
            "categorical_columns": llm_out["categorical_columns"],
            "entities": llm_out["entities"],
        }

        # Write per-table output
        clean_path = os.path.join(TABLES_CLEAN_DIR, f"{table_id}.json")
        save_json(clean_path, clean_obj)

        # Add to combined summary
        all_schema[table_id] = {
            "domain": domain,
            "title": title,
            "columns": columns_original,
            "description": llm_out["description"],
            "numeric_columns": llm_out["numeric_columns"],
            "categorical_columns": llm_out["categorical_columns"],
            "entities": llm_out["entities"],
        }

    # Save combined file
    save_json(SCHEMA_DESCRIPTIONS_PATH, all_schema)

    print(f"\n Wrote per-table clean files to: {TABLES_CLEAN_DIR}/")
    print(f" Wrote combined schema file to: {SCHEMA_DESCRIPTIONS_PATH}")


if __name__ == "__main__":
    main()