"""
Experiment: Cluster tables using HDBSCAN directly on raw embeddings (no UMAP).

Same as src/cluster.py but skips UMAP dimensionality reduction.
HDBSCAN runs on raw 384-dim embeddings using cosine metric
(euclidean distances are unreliable in high dimensions).

Run alongside the baseline (src/cluster.py) to compare:
  - Number of clusters formed
  - Noise points (topic -1)
  - Silhouette score
  - Qualitative cluster coherence
"""

import json
import argparse
import os
import numpy as np
from collections import defaultdict, Counter
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score


class IdentityReducer(BaseEstimator, TransformerMixin):
    """Drop-in UMAP replacement that passes embeddings through unchanged."""

    def fit_transform(self, X, y=None):
        self.embedding_ = X
        return X

    def fit(self, X, y=None):
        self.embedding_ = X
        return self

    def transform(self, X):
        return X


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
    print("NOTE: Running WITHOUT UMAP — HDBSCAN operates on L2-normalized embeddings (euclidean ≈ cosine)")

    # Load extra metadata from tables_clean/ if available
    tables_meta = load_tables_clean(args.tables_dir)
    if tables_meta:
        print(f"Loaded metadata for {len(tables_meta)} tables from {args.tables_dir}")

    # Extract arrays for BERTopic
    table_ids = [t["table_id"] for t in tables_data]
    docs = [t["description"] for t in tables_data]
    embeddings = np.array([t["embedding"] for t in tables_data])

    # L2-normalize so euclidean distance == cosine distance (avoids metric support issues)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / np.where(norms == 0, 1, norms)

    # No UMAP — pass embeddings through unchanged
    identity_reducer = IdentityReducer()

    # euclidean on L2-normalized vectors is equivalent to cosine distance
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        metric="euclidean",
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
        umap_model=identity_reducer,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=args.min_topic_size,
        top_n_words=10,
        verbose=True,
        low_memory=False,
    )

    # Cluster — pass pre-computed embeddings, skip SentenceTransformer
    print("\nRunning BERTopic clustering (no UMAP)...")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings_norm)

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
    print("CLUSTERING RESULTS (no UMAP)")
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

    # Compute silhouette score on raw embeddings (cosine) for non-noise topics
    topic_labels = np.array(topics)
    valid_mask = topic_labels != -1

    metrics = {
        "n_tables": len(table_ids),
        "n_topics": len(set(topics)) - (1 if -1 in topics else 0),
        "n_noise": noise_count,
        "cluster_sizes": dict(Counter(topics)),
        "umap_used": False,
    }

    if valid_mask.sum() >= 2 and len(set(topic_labels[valid_mask])) >= 2:
        X = embeddings_norm[valid_mask]
        y = topic_labels[valid_mask]
        sil = silhouette_score(X, y, metric="cosine")
        metrics["silhouette_score"] = round(float(sil), 4)
        metrics["silhouette_metric"] = "cosine on raw 384-dim embeddings"
        print(f"\nSilhouette score (cosine, raw embeddings): {sil:.4f}")
        print("Note: not directly comparable to UMAP-reduced silhouette score")

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
    parser = argparse.ArgumentParser(
        description="Cluster tables using HDBSCAN on raw embeddings (no UMAP)"
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to pre-computed embeddings JSON")
    parser.add_argument("--tables_dir", type=str, default=None,
                        help="Path to tables_clean/ dir for extra metadata (title, domain)")
    parser.add_argument("--output_dir", type=str, default="data/output_no_umap")

    # HDBSCAN params — same defaults as baseline for fair comparison
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=3)
    parser.add_argument("--hdbscan_epsilon", type=float, default=0.0)

    # BERTopic params
    parser.add_argument("--min_topic_size", type=int, default=3)
    parser.add_argument("--vectorizer_min_df", type=int, default=2)

    args = parser.parse_args()
    main(args)
