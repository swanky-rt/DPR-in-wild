#!/usr/bin/env python3
"""
Layer 1B: Generate table-level embeddings using SentenceTransformer.

INPUT:
- tables_clean/T1.json ... tables_clean/T100.json
  (These files are expected to be produced by Layer 1A.)

OUTPUT:
- table_embeddings.json
  A single JSON list with one object per table:
    {
      "table_id": "...",
      "columns": [...],          # preserved column names
      "description": "...",      # LLM-generated description from Layer 1A
      "embedding": [...]         # embedding vector for clustering
    }

Embedding text per table:
- For each table, build one document string:
  "Title: <title>. Columns: <col1, col2, ...>. Description: <description>"
- This combines table meaning (description) with schema cues (column names).
"""

import os
import json
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer


# ----------------------------
# Configuration
# ----------------------------

# A commonly used, lightweight sentence embedding model suitable for clustering baselines.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Directory containing Layer 1A outputs.
TABLES_CLEAN_DIR = "tables_clean"

# Output JSON file containing embeddings for all tables.
OUT_PATH = "table_embeddings.json"


# ----------------------------
# File utilities
# ----------------------------

def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return it as a Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    """Save a Python object as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------
# Embedding text construction
# ----------------------------

def build_document(title: str, columns: List[str], description: str) -> str:
    """
    Build the single text document that will be embedded for one table.

    The document includes:
    - Title (if available)
    - Full list of column names (preserved)
    - The table description (from Layer 1A)
    """
    title = title.strip() if title else ""
    description = description.strip() if description else ""
    columns_str = ", ".join(columns) if columns else ""

    return f"Title: {title}. Columns: {columns_str}. Description: {description}"


# ----------------------------
# Main pipeline
# ----------------------------

def main() -> None:
    """
    Load the SentenceTransformer model (downloads once if not cached).
    For each table file T1..T100 in tables_clean/:
       - Read title, columns, description
       - Build a document string (title + columns + description)
       - Generate an embedding vector for that document
       - Store the result in a list
    Write all results to table_embeddings.json
    """
    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    results: List[Dict[str, Any]] = []

    for i in range(1, 101):
        table_id = f"T{i}"
        input_path = os.path.join(TABLES_CLEAN_DIR, f"{table_id}.json")

        if not os.path.exists(input_path):
            print(f"[WARN] Missing input file: {input_path}. Skipping.")
            continue

        table_obj = load_json(input_path)

        title = table_obj.get("title", "")
        columns = table_obj.get("columns", [])
        description = table_obj.get("description", "")

        document = build_document(title=title, columns=columns, description=description)

        # Generate one embedding vector per table document.
        embedding_vec = model.encode(document, convert_to_numpy=True)

        results.append({
            "table_id": table_id,
            "columns": columns,
            "description": description,
            "embedding": embedding_vec.tolist()
        })

        print(f"[OK] Embedded {table_id} (dim={len(embedding_vec)})")

    save_json(OUT_PATH, results)
    print(f"\n[OK] Wrote embeddings file: {OUT_PATH}")


if __name__ == "__main__":
    main()