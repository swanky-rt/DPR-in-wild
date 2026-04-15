"""
Summary:
- Loads the HybridQA train split from Hugging Face
- Picks 100 unique tables while limiting how many come from the same domain
- Renames the chosen tables to T1..T100 for consistent local use
- Saves the tables to tables_raw/ and writes a table_manifest.json with basic stats
"""

import os
import json
from collections import Counter
from typing import Dict, Any, List, Tuple

from datasets import load_dataset


# -----------------------------
#  Domain guessing (heuristic)
# -----------------------------

def guess_domain_from_title(table_title: str) -> str:
    """
    Guess a coarse domain label from the table title using simple keyword matching.
    """
    # If the title is empty/missing, we can’t guess a domain
    if not table_title:
        return "Unknown"

    # Convert to lowercase so matching is case-insensitive
    title_lower = table_title.lower()

    # Keyword buckets for a rough domain guess
    domain_keywords = {
        "Sports":     ["football", "soccer", "basketball", "baseball", "tennis", "olympic", "league", "championship"],
        "Music":      ["album", "song", "band", "music", "singer", "composer"],
        "Film/TV":    ["film", "movie", "television", "tv", "actor", "director", "episode"],
        "Geography":  ["city", "country", "province", "river", "mountain", "island", "county"],
        "Politics":   ["election", "minister", "president", "parliament", "senator", "governor"],
        "Science":    ["species", "chemical", "biology", "physics", "astronomy", "planet", "genus"],
    }

    # Return the first matching domain
    for domain_name, keywords in domain_keywords.items():
        if any(keyword in title_lower for keyword in keywords):
            return domain_name

    # If nothing matches, label it as General
    return "General"


# --------------------------------
#  Table size stats (rows / cols)
# --------------------------------

def get_table_shape(hybridqa_table: Dict[str, Any]) -> Tuple[int, int]:
    """
    Compute basic table shape (num rows, num columns) from a HybridQA table object.
    """
    # HybridQA stores column names in "header" and rows in "data"
    column_headers = hybridqa_table.get("header") or []
    row_data = hybridqa_table.get("data") or []

    num_rows = len(row_data)
    num_columns = len(column_headers)
    return num_rows, num_columns


# -----------------------------------------------
#  Build an index: table_id -> first row index
# -----------------------------------------------

def build_first_row_index_by_table_id(dataset) -> Dict[str, int]:
    """
    Build a mapping from each table_id to the first dataset row index where it appears.
    This makes it fast to retrieve a table without scanning the dataset repeatedly.
    """
    first_index_for_table_id: Dict[str, int] = {}

    for row_index, example in enumerate(dataset):
        table_id = example["table_id"]

        # Store only the first occurrence of each table_id
        if table_id not in first_index_for_table_id:
            first_index_for_table_id[table_id] = row_index

    return first_index_for_table_id


# --------------------------------------------
#  Select 100 tables with light domain diversity
# --------------------------------------------

def select_diverse_tables(
    dataset,
    num_tables_to_select: int = 100,
    max_tables_per_domain: int = 3
) -> List[Dict[str, Any]]:
    """
    Select tables while limiting how many come from the same guessed domain.
    First pass: enforce per-domain quota.
    Second pass: fill remaining slots if we’re still short.
    """
    # Get all unique table_ids in a stable (sorted) order
    unique_table_ids: List[str] = sorted(set(dataset["table_id"]))

    # Precompute where each table_id first shows up
    first_row_index_by_table_id = build_first_row_index_by_table_id(dataset)

    selected_tables: List[Dict[str, Any]] = []
    domain_counts = Counter()

    # First pass: respect the domain quota
    for source_table_id in unique_table_ids:
        first_row_index = first_row_index_by_table_id[source_table_id]
        example_row = dataset[first_row_index]
        table_obj = example_row["table"]

        table_title = table_obj.get("title") or ""
        guessed_domain = guess_domain_from_title(table_title)

        # Skip if we already picked enough tables from this domain
        if domain_counts[guessed_domain] >= max_tables_per_domain:
            continue

        selected_tables.append({
            "source_table_id": source_table_id,
            "domain": guessed_domain,
            "table_obj": table_obj,
        })
        domain_counts[guessed_domain] += 1

        # Stop when we have enough tables
        if len(selected_tables) >= num_tables_to_select:
            break

    # Second pass: if still short, fill remaining ignoring the quota
    if len(selected_tables) < num_tables_to_select:
        already_selected_ids = {x["source_table_id"] for x in selected_tables}

        for source_table_id in unique_table_ids:
            if len(selected_tables) >= num_tables_to_select:
                break
            if source_table_id in already_selected_ids:
                continue

            first_row_index = first_row_index_by_table_id[source_table_id]
            example_row = dataset[first_row_index]
            table_obj = example_row["table"]

            table_title = table_obj.get("title") or ""
            guessed_domain = guess_domain_from_title(table_title)

            selected_tables.append({
                "source_table_id": source_table_id,
                "domain": guessed_domain,
                "table_obj": table_obj,
            })

    return selected_tables


# -----------------------
#  Main program function
# -----------------------

def main() -> None:
    """
    Load data, select tables, write table JSON files + a manifest, and print a summary.
    """
    
    script_dir = os.path.dirname(__file__)

   
    repo_root = script_dir

  
    output_tables_dir = os.path.join(repo_root, "tables_raw")
    output_manifest_path = os.path.join(repo_root, "table_manifest.json")

    # Ensure output dir exists
    os.makedirs(output_tables_dir, exist_ok=True)

    print("Loading HybridQA from Hugging Face...")
    dataset = load_dataset("wenhu/hybrid_qa", split="train")
    print(f"Loaded dataset with {len(dataset)} QA examples.")

    print("Selecting 100 tables with domain diversity constraint...")
    selected = select_diverse_tables(
        dataset=dataset,
        num_tables_to_select=100,
        max_tables_per_domain=3
    )

    manifest_entries: List[Dict[str, Any]] = []

    # Write each selected table as T1..T100
    for normalized_index, selection in enumerate(selected, start=1):
        normalized_table_id = f"T{normalized_index}"

        source_table_id = selection["source_table_id"]
        domain = selection["domain"]
        table_obj = selection["table_obj"]

        num_rows, num_columns = get_table_shape(table_obj)

        # Path for this table file
        output_table_file_path = os.path.join(output_tables_dir, f"{normalized_table_id}.json")

        # Write raw table JSON file
        with open(output_table_file_path, "w", encoding="utf-8") as out_f:
            json.dump(
                {
                    "table_id": normalized_table_id,      # normalized ID T1..T100
                    "source": "HybridQA",                 # dataset source label
                    "source_table_id": source_table_id,   # original HybridQA table_id for traceability
                    "domain": domain,                     # guessed domain
                    "table": table_obj,                   # raw table content (title/header/data/...)
                },
                out_f,
                indent=2,
                ensure_ascii=False
            )

        # Add entry to manifest
        manifest_entries.append({
            "table_id": normalized_table_id,
            "source": "HybridQA",
            "domain": domain,
            "num_rows": num_rows,
            "num_columns": num_columns,
            "path": f"tables_raw/{normalized_table_id}.json",
        })

    # Write manifest JSON file
    with open(output_manifest_path, "w", encoding="utf-8") as manifest_f:
        json.dump(manifest_entries, manifest_f, indent=2, ensure_ascii=False)

   
    print("\nDONE.")
    print(f"Wrote 100 table files to: {output_tables_dir}")
    print(f"Wrote manifest to: {output_manifest_path}")

    print("\nSelected tables summary:")
    for entry in manifest_entries:
        print(
            f"{entry['table_id']}: "
            f"domain={entry['domain']}, "
            f"rows={entry['num_rows']}, "
            f"cols={entry['num_columns']}, "
            f"path={entry['path']}"
        )




if __name__ == "__main__":
    main()