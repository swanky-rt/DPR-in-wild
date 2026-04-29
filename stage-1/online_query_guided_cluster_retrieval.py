#!/usr/bin/env python3
"""
Online query-guided cluster retrieval using offline cluster output.

This script takes a set of user queries and finds the offline clusters
most relevant to each query.

For each query, the script:
- encodes the query into an embedding vector
- computes cosine similarity between the query embedding and each table embedding
- keeps the most relevant tables that pass a similarity threshold
- finds clusters that contain at least one of those matched tables

Input files:
- table_embeddings.json
- tables_clean/T1.json ... T100.json
- clusters_summary.json

Output file:
- query_table_cluster_matches.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Defaults (overridden by CLI args)
# --------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

DEFAULT_TABLE_EMBEDDINGS_PATH = str(_SCRIPT_DIR / "table_embeddings.json")
DEFAULT_TABLES_CLEAN_DIR = str(_SCRIPT_DIR / "tables_clean")
DEFAULT_CLUSTERS_SUMMARY_PATH = str(_REPO_ROOT / "stage-2" / "data" / "output_qwen_emb_v2" / "clusters_summary.json")
DEFAULT_OUTPUT_PATH = str(_SCRIPT_DIR / "query_table_cluster_matches.json")
DEFAULT_TOP_K_TABLES = 15
DEFAULT_SIMILARITY_THRESHOLD = 0.4
DEFAULT_INCLUDE_NOISE_CLUSTER = False

# Fallback queries used when no --queries_file is provided
_DEFAULT_QUERIES = [
    {
        "query_id": "Q1",
        "query_text": "Who won the 1913 International Cross-Country Championships?"
    },
    {
        "query_id": "Q2",
        "query_text": "Show which athletes or race records appear across championship result tables and yearly performance tables, and summarize their rankings, times, and event context."
    },
    {
        "query_id": "Q3",
        "query_text": "Compare how success is measured in the sports tables and the movie tables, using rankings and times for athletes and revenue, admissions, or gross for films."
    }
]


def _load_queries(queries_file: str) -> List[Dict[str, str]]:
    """
    Load user queries from a JSON file.
    Supports two formats:
      - [{"query_id": "Q1", "query_text": "..."}, ...]
      - [{"dpr_id": "Q1", "user_query": "..."}, ...]  (stage-2 user_queries format)
    """
    with open(queries_file, encoding="utf-8") as f:
        raw = json.load(f)
    queries = []
    for i, entry in enumerate(raw):
        query_id = entry.get("query_id") or entry.get("dpr_id") or f"Q{i+1}"
        query_text = entry.get("query_text") or entry.get("user_query") or ""
        if query_text:
            queries.append({"query_id": query_id, "query_text": query_text})
    return queries


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def load_json(path: str) -> Any:
    """Read and return the contents of a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    """Save a Python object to a JSON file with readable formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_table_metadata(table_id: str, tables_clean_dir: str) -> Dict[str, Any]:
    """
    Load metadata for one table from tables_clean/T?.json.

    If the file is missing, return empty placeholder values so the script
    can continue running.
    """
    path = os.path.join(tables_clean_dir, f"{table_id}.json")

    if not os.path.exists(path):
        return {
            "title": "",
            "columns": [],
            "description": ""
        }

    obj = load_json(path)
    return {
        "title": obj.get("title", ""),
        "columns": obj.get("columns", []),
        "description": obj.get("description", "")
    }


def load_table_embeddings(embeddings_path: str, tables_clean_dir: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Load precomputed table embeddings and attach basic metadata.

    Returns:
    - table_records: list of dictionaries containing table_id, title,
      columns, and description
    - table_embeddings: numpy array of shape [num_tables, embedding_dim]
    """
    raw = load_json(embeddings_path)

    table_records: List[Dict[str, Any]] = []
    embeddings: List[List[float]] = []

    for item in raw:
        table_id = item["table_id"]
        meta = load_table_metadata(table_id, tables_clean_dir)

        table_records.append({
            "table_id": table_id,
            "title": meta.get("title", ""),
            "columns": meta.get("columns", []),
            "description": meta.get("description", item.get("description", ""))
        })
        embeddings.append(item["embedding"])

    return table_records, np.array(embeddings, dtype=np.float32)


def load_offline_clusters_from_summary(summary_path: str, include_noise: bool = False) -> List[Dict[str, Any]]:
    """
    Read the offline cluster summary file and convert it into a simpler
    structure used by this script.

    Each returned cluster contains:
    - cluster_id
    - topic_id
    - theme
    - table_ids
    """
    obj = load_json(summary_path)

    # Map each topic id to its readable topic name
    topic_name_map: Dict[str, str] = {}
    for item in obj.get("topic_info", []):
        topic_id = str(item["Topic"])
        topic_name_map[topic_id] = item.get("Name", f"topic_{topic_id}")

    offline_clusters: List[Dict[str, Any]] = []
    next_cluster_id = 1

    for cluster in obj.get("clusters", []):
        topic_id = str(cluster["topic_id"])

        # Skip the noise cluster 
        if not include_noise and topic_id == "-1":
            continue

        offline_clusters.append({
            "cluster_id": next_cluster_id,
            "topic_id": topic_id,
            "theme": topic_name_map.get(topic_id, f"topic_{topic_id}"),
            "table_ids": cluster["table_ids"]
        })
        next_cluster_id += 1

    return offline_clusters


def build_table_to_cluster_map(clusters: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a lookup from each table id to the list of clusters that contain it.

    Example:
    T6  -> [cluster_info]
    T22 -> [cluster_info]
    """
    table_to_clusters: Dict[str, List[Dict[str, Any]]] = {}

    for cluster in clusters:
        for table_id in cluster["table_ids"]:
            table_to_clusters.setdefault(table_id, []).append(cluster)

    return table_to_clusters


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match user queries to offline BERTopic clusters via embedding similarity."
    )
    parser.add_argument("--embeddings_path", default=DEFAULT_TABLE_EMBEDDINGS_PATH,
                        help="Path to table_embeddings.json")
    parser.add_argument("--tables_clean_dir", default=DEFAULT_TABLES_CLEAN_DIR,
                        help="Path to tables_clean/ directory")
    parser.add_argument("--clusters_summary_path", default=DEFAULT_CLUSTERS_SUMMARY_PATH,
                        help="Path to clusters_summary.json from stage-2 clustering")
    parser.add_argument("--queries_file", default=None,
                        help="JSON file of user queries. If omitted, uses built-in default queries.")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH,
                        help="Where to write query_cluster_matches.json")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K_TABLES,
                        help="Max tables to retrieve per query (default: 15)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                        help="Min cosine similarity to keep a table (default: 0.4)")
    parser.add_argument("--include_noise", action="store_true",
                        help="Include the noise cluster (-1) in matches")
    args = parser.parse_args()

    user_queries = _load_queries(args.queries_file) if args.queries_file else _DEFAULT_QUERIES

    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("[INFO] Loading table embeddings...")
    table_records, table_embeddings = load_table_embeddings(args.embeddings_path, args.tables_clean_dir)
    print(f"[INFO] Loaded {len(table_records)} tables with shape {table_embeddings.shape}")

    print(f"[INFO] Loading offline cluster summary from: {args.clusters_summary_path}")
    offline_clusters = load_offline_clusters_from_summary(
        args.clusters_summary_path,
        include_noise=args.include_noise
    )
    print(f"[INFO] Loaded {len(offline_clusters)} usable clusters")

    # Build a lookup so matched tables can be mapped to clusters efficiently
    table_to_clusters = build_table_to_cluster_map(offline_clusters)

    print("[INFO] Embedding user queries...")
    query_texts = [q["query_text"] for q in user_queries]
    query_embeddings = model.encode(query_texts, convert_to_numpy=True)

    all_results = []

    for i, query_obj in enumerate(user_queries):
        query_id = query_obj["query_id"]
        query_text = query_obj["query_text"]
        query_embedding = query_embeddings[i].reshape(1, -1)

        # Compute cosine similarity between the current query embedding
        # and every table embedding. Higher cosine similarity means the
        # query and table are more closely aligned in embedding space.
        sims = cosine_similarity(query_embedding, table_embeddings)[0]

        # Keep only tables whose cosine similarity score is at least
        # the configured threshold.
        candidate_indices = [
            idx for idx, sim in enumerate(sims)
            if sim >= args.threshold
        ]

        # Sort the remaining candidate tables by cosine similarity,
        # from highest score to lowest score.
        candidate_indices = sorted(
            candidate_indices,
            key=lambda idx: sims[idx],
            reverse=True
        )

        # Keep only the top-k highest-scoring tables.
        top_indices = candidate_indices[:args.top_k]

        matched_tables = []
        matched_table_ids = []

        for idx in top_indices:
            record = table_records[idx]
            table_id = record["table_id"]
            matched_table_ids.append(table_id)

            matched_tables.append({
                "table_id": table_id,
                "title": record["title"],
                "similarity": round(float(sims[idx]), 4),
                "columns": record["columns"]
            })

        # Find clusters that contain at least one of those matched tables.
        matched_clusters: Dict[int, Dict[str, Any]] = {}

        for table_id in matched_table_ids:
            clusters_for_table = table_to_clusters.get(table_id, [])

            for cluster in clusters_for_table:
                cid = cluster["cluster_id"]

                if cid not in matched_clusters:
                    matched_clusters[cid] = {
                        "cluster_id": cid,
                        "topic_id": cluster["topic_id"],
                        "theme": cluster["theme"],
                        "matched_tables_in_cluster": [],
                        "all_tables_in_cluster": cluster["table_ids"],
                        "num_tables_in_cluster": len(cluster["table_ids"])
                    }

                matched_clusters[cid]["matched_tables_in_cluster"].append(table_id)

        # Rank matched clusters by how many retrieved tables they contain.
        matched_clusters_list = list(matched_clusters.values())
        matched_clusters_list = sorted(
            matched_clusters_list,
            key=lambda x: len(x["matched_tables_in_cluster"]),
            reverse=True
        )

        all_results.append({
            "query_id": query_id,
            "query_text": query_text,
            "num_tables_above_threshold": len(candidate_indices),
            "matched_tables": matched_tables,
            "matched_clusters": matched_clusters_list
        })

    output = {
        "config": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "top_k_tables": args.top_k,
            "similarity_threshold": args.threshold,
            "clusters_summary_path": args.clusters_summary_path,
            "include_noise_cluster": args.include_noise,
        },
        "query_results": all_results
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    save_json(args.output_path, output)

    print(f"\n[OK] Wrote output to: {args.output_path}")
    print("\n================ RESULTS ================\n")

    for item in all_results:
        print(f'{item["query_id"]}: {item["query_text"]}')
        print(f'Tables above threshold: {item["num_tables_above_threshold"]}')

        print("\nTop matched tables:")
        if not item["matched_tables"]:
            print("  - None passed the threshold")
        else:
            for t in item["matched_tables"]:
                print(f'  - {t["table_id"]} | title="{t["title"]}" | similarity={t["similarity"]}')

        print("\nClusters containing at least one matched table:")
        if not item["matched_clusters"]:
            print("  - None")
        else:
            for c in item["matched_clusters"]:
                print(
                    f'  - Cluster {c["cluster_id"]} (topic {c["topic_id"]}) | '
                    f'{c["theme"]} | matched tables: {sorted(set(c["matched_tables_in_cluster"]))}'
                )
        print("-" * 70)


if __name__ == "__main__":
    main()