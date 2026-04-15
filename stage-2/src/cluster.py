"""
Step 2: Cluster tables using BERTopic.

Takes pre-computed embeddings JSON (table_id, columns, description, embedding)
and clusters semantically similar tables using BERTopic
(UMAP + HDBSCAN + CountVectorizer).

Pre-computed embeddings are passed directly to BERTopic — no
SentenceTransformer encoding happens here.

Adapted from DPBench benchmark_framework/src/cluster.py.
"""

import json
import argparse
import os
import numpy as np
from collections import defaultdict, Counter
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score


def load_embeddings(path):
    """Load pre-computed embeddings JSON (list of dicts)."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_tables_clean(tables_dir):
    """Load extra metadata (title, domain, etc.) from tables_clean/ JSONs."""
    metadata = {}
    if not tables_dir or not os.path.isdir(tables_dir):
        return metadata
    for fname in os.listdir(tables_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(tables_dir, fname)
        with open(fpath, "r") as f:
            table = json.load(f)
        tid = table.get("table_id", fname.replace(".json", ""))
        metadata[tid] = table
    return metadata


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pre-computed embeddings
    print(f"Loading embeddings from {args.input_path}...")
    tables_data = load_embeddings(args.input_path)
    print(f"Loaded {len(tables_data)} tables with {len(tables_data[0]['embedding'])}-dim embeddings")

    # Load extra metadata from tables_clean/ if available
    tables_meta = load_tables_clean(args.tables_dir)
    if tables_meta:
        print(f"Loaded metadata for {len(tables_meta)} tables from {args.tables_dir}")

    # Extract arrays for BERTopic
    table_ids = [t["table_id"] for t in tables_data]
    docs = [t["description"] for t in tables_data]
    embeddings = np.array([t["embedding"] for t in tables_data])

    # Build BERTopic components
    umap_model = UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        n_components=args.umap_n_components,
        metric=args.umap_metric,
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        metric=args.hdbscan_metric,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=args.hdbscan_epsilon,
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=args.vectorizer_min_df,
        ngram_range=(1, 2),
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=args.min_topic_size,
        top_n_words=10,
        verbose=True,
        low_memory=False,
    )

    # Cluster — pass pre-computed embeddings, skip SentenceTransformer
    print("\nRunning BERTopic clustering...")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # Build cluster output
    clustered_tables = defaultdict(list)
    for idx, topic in enumerate(topics):
        tid = table_ids[idx]
        meta = tables_meta.get(tid, {})

        table_info = {
            "table_id": tid,
            "title": meta.get("title", ""),
            "domain": meta.get("domain", ""),
            "columns": tables_data[idx]["columns"],
            "description": tables_data[idx]["description"],
        }
        clustered_tables[str(topic)].append(table_info)

    # Print results
    print(f"\n{'='*60}")
    print("CLUSTERING RESULTS")
    print(f"{'='*60}")
    print(f"Total tables: {len(table_ids)}")
    print(f"Total topics: {len(clustered_tables)}")

    noise_count = len(clustered_tables.get("-1", []))
    if noise_count:
        print(f"Noise (topic -1): {noise_count} tables")

    for cid, tables in sorted(clustered_tables.items(), key=lambda x: int(x[0])):
        label = "NOISE" if cid == "-1" else f"Topic {cid}"
        print(f"\n--- {label} ({len(tables)} tables) ---")
        for t in tables:
            title_str = f" — {t['title']}" if t['title'] else ""
            print(f"  [{t['table_id']}]{title_str}")
            print(f"    Columns: {', '.join(t['columns'][:6])}{'...' if len(t['columns']) > 6 else ''}")
            print(f"    Desc: {t['description'][:100]}...")

    # Compute metrics on non-noise topics
    topic_labels = np.array(topics)
    valid_mask = topic_labels != -1

    metrics = {
        "n_tables": len(table_ids),
        "n_topics": len(set(topics)) - (1 if -1 in topics else 0),
        "n_noise": noise_count,
        "cluster_sizes": dict(Counter(topics)),
    }

    if valid_mask.sum() >= 2 and len(set(topic_labels[valid_mask])) >= 2:
        umap_embeds = topic_model.umap_model.embedding_
        X = umap_embeds[valid_mask]
        y = topic_labels[valid_mask]
        finite_mask = (~np.isnan(X).any(axis=1)) & (~np.isinf(X).any(axis=1))
        if finite_mask.sum() >= 2:
            sil = silhouette_score(X[finite_mask], y[finite_mask], metric="cosine")
            metrics["silhouette_score"] = round(float(sil), 4)
            print(f"\nSilhouette score: {sil:.4f}")

    # Save clusters
    clusters_path = os.path.join(args.output_dir, "clusters.json")
    with open(clusters_path, "w") as f:
        json.dump(clustered_tables, f, indent=2)
    print(f"\nSaved clusters to {clusters_path}")

    # Save summary
    summary = {
        "metrics": metrics,
        "topic_info": topic_model.get_topic_info().to_dict(orient="records"),
        "clusters": [
            {
                "topic_id": cid,
                "num_tables": len(tables),
                "table_ids": [t["table_id"] for t in tables],
            }
            for cid, tables in sorted(clustered_tables.items(), key=lambda x: int(x[0]))
        ],
    }
    summary_path = os.path.join(args.output_dir, "clusters_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster tables using BERTopic")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to pre-computed embeddings JSON")
    parser.add_argument("--tables_dir", type=str, default=None,
                        help="Path to tables_clean/ dir for extra metadata (title, domain)")
    parser.add_argument("--output_dir", type=str, default="data/output")

    # UMAP params — tuned for ~100 tables
    # n_neighbors ~= sqrt(n_tables); n_components=10 gives better separation at this scale
    parser.add_argument("--umap_n_neighbors", type=int, default=10)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_n_components", type=int, default=10)
    parser.add_argument("--umap_metric", type=str, default="cosine")

    # HDBSCAN params — tuned for ~100 tables
    # min_cluster_size=3 allows more granular clusters for higher DPR count
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=3)
    parser.add_argument("--hdbscan_metric", type=str, default="euclidean")
    parser.add_argument("--hdbscan_epsilon", type=float, default=0.0)

    # BERTopic params — tuned for ~100 tables
    parser.add_argument("--min_topic_size", type=int, default=3)
    parser.add_argument("--vectorizer_min_df", type=int, default=2)

    args = parser.parse_args()
    main(args)
