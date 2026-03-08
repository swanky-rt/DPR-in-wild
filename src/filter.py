"""
Step 2b: Filter and refine clusters.

- Drops noise cluster (topic -1 from BERTopic/HDBSCAN)
- Drops single-table clusters (need >= 2 tables for a meaningful DPR)
- Splits oversized clusters using KMeans on schema embeddings
- Assigns sequential dpr_ids

Adapted from DPBench filter.py.
"""

import json
import argparse
import os
import numpy as np
from math import ceil
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def build_schema_text(table_info):
    """Build text representation for schema embedding."""
    title = table_info.get("title", "")
    columns = " | ".join(table_info["columns"])
    desc = table_info.get("description", "")
    return f"{title}. Columns: {columns}. {desc}"


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.clusters_path, "r") as f:
        raw_clusters = json.load(f)

    print(f"Loaded {len(raw_clusters)} raw clusters (including noise)")

    model = None
    filtered = []
    dropped_noise = 0
    dropped_small = 0

    for cid, tables in raw_clusters.items():
        # Drop noise cluster
        if cid == "-1":
            dropped_noise = len(tables)
            continue

        # Drop single-table clusters
        if len(tables) < args.min_tables:
            dropped_small += len(tables)
            continue

        if len(tables) <= args.max_tables:
            filtered.append((cid, tables))
        else:
            # Split oversized cluster using KMeans
            if model is None:
                print("Loading embedding model for splitting large clusters...")
                model = SentenceTransformer(args.embedding_model)

            schema_texts = [build_schema_text(t) for t in tables]
            embeds = model.encode(schema_texts, show_progress_bar=False)

            n_sub = ceil(len(tables) / args.max_tables)
            km = KMeans(n_clusters=n_sub, random_state=42, n_init="auto")
            labels = km.fit_predict(embeds)

            groups = defaultdict(list)
            for lbl, tbl in zip(labels, tables):
                groups[lbl].append(tbl)

            for lbl, group in groups.items():
                if len(group) >= args.min_tables:
                    filtered.append((f"{cid}_sub{lbl}", group))

    # Build output with sequential dpr_ids
    output = []
    for idx, (cluster_key, tables) in enumerate(filtered, start=1):
        output.append({
            "dpr_id": str(idx),
            "cluster_key": cluster_key,
            "tables": tables,
        })

    # Save
    output_path = os.path.join(args.output_dir, "filtered_clusters.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults:")
    print(f"  Noise dropped: {dropped_noise} tables")
    print(f"  Small clusters dropped: {dropped_small} tables")
    print(f"  Kept: {len(output)} clusters")
    for item in output:
        tids = [t["table_id"] for t in item["tables"]]
        print(f"    DPR {item['dpr_id']} (topic {item['cluster_key']}): {len(tids)} tables — {tids}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and refine clusters")
    parser.add_argument("--clusters_path", type=str, required=True,
                        help="Path to clusters.json from clustering step")
    parser.add_argument("--output_dir", type=str, default="data/output")
    parser.add_argument("--min_tables", type=int, default=2)
    parser.add_argument("--max_tables", type=int, default=30)
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    main(args)
