"""
Cluster Quality Evaluation

Reads pre-computed embeddings and clustering output and computes a comprehensive
set of cluster quality metrics inspired by the IBM Research DPR proposal.

Metric groups:
  1. Geometric  — Silhouette, Davies-Bouldin, Calinski-Harabasz
  2. Semantic   — Intra-cluster cosine similarity, inter-cluster centroid distance
  3. Schema     — Jaccard similarity of column name sets within clusters
  4. Health     — Noise rate, size distribution, singleton/doubleton rate

Usage:
    python src/evaluate_clusters.py \
        --embeddings_path data/input_100tables/qwen_table_embeddings.json \
        --clusters_summary_path data/output_qwen_emb_v2/clusters_summary.json \
        --output_dir data/output_qwen_emb_v2
"""

import json
import argparse
import os
import numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize


# ── Utilities ────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def cosine_similarity(a, b):
    """Cosine similarity between two 1-D numpy vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


# ── Metric Calculators ────────────────────────────────────────────────────────

def geometric_metrics(embeddings_matrix, labels):
    """
    Sklearn geometric cluster quality metrics.
    Computed on the raw (high-dim) embeddings, not UMAP-reduced space.
    """
    valid_mask = labels != -1
    X = embeddings_matrix[valid_mask]
    y = labels[valid_mask]

    if len(set(y)) < 2 or len(X) < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
                "note": "Not enough clusters for geometric metrics"}

    sil   = float(silhouette_score(X, y, metric="cosine"))
    db    = float(davies_bouldin_score(X, y))
    ch    = float(calinski_harabasz_score(X, y))

    return {
        "silhouette_score": round(sil, 4),       # higher is better (-1 to 1)
        "davies_bouldin_index": round(db, 4),    # lower is better (>= 0)
        "calinski_harabasz_index": round(ch, 2), # higher is better
    }


def intra_cluster_coherence(cluster_embeddings_map):
    """
    For each cluster: mean pairwise cosine similarity of table embeddings.
    Returns per-cluster values + aggregate mean.
    """
    per_cluster = {}
    for cid, embs in cluster_embeddings_map.items():
        if len(embs) < 2:
            per_cluster[cid] = {"n_tables": len(embs), "avg_cosine_sim": None,
                                 "note": "single table"}
            continue
        sims = [cosine_similarity(a, b)
                for a, b in combinations(embs, 2)]
        per_cluster[cid] = {
            "n_tables": len(embs),
            "avg_cosine_sim": round(float(np.mean(sims)), 4),
            "min_cosine_sim": round(float(np.min(sims)), 4),
        }

    valid_scores = [v["avg_cosine_sim"] for v in per_cluster.values()
                    if v.get("avg_cosine_sim") is not None]
    aggregate = round(float(np.mean(valid_scores)), 4) if valid_scores else None
    return {"per_cluster": per_cluster, "mean_intra_cluster_cosine_sim": aggregate}


def inter_cluster_separation(cluster_embeddings_map):
    """
    Centroid-vs-centroid cosine distances between all cluster pairs.
    Higher mean distance = better-separated clusters.
    """
    centroids = {cid: np.mean(embs, axis=0)
                 for cid, embs in cluster_embeddings_map.items() if embs}

    if len(centroids) < 2:
        return {"mean_centroid_cosine_distance": None,
                "note": "Not enough clusters"}

    cids = list(centroids.keys())
    distances = []
    for i, j in combinations(range(len(cids)), 2):
        sim = cosine_similarity(centroids[cids[i]], centroids[cids[j]])
        distances.append(1.0 - sim)  # cosine distance

    return {
        "mean_centroid_cosine_distance": round(float(np.mean(distances)), 4),  # higher = better
        "min_centroid_cosine_distance":  round(float(np.min(distances)), 4),
        "n_cluster_pairs": len(distances),
    }


def schema_overlap_metrics(cluster_columns_map):
    """
    Within each cluster, compute mean pairwise Jaccard similarity of column name sets.
    High intra-cluster Jaccard = tables share many column names (strong schema overlap).
    """
    per_cluster = {}
    for cid, col_sets in cluster_columns_map.items():
        if len(col_sets) < 2:
            per_cluster[cid] = {"n_tables": len(col_sets), "avg_jaccard": None,
                                 "note": "single table"}
            continue
        sims = [jaccard(a, b) for a, b in combinations(col_sets, 2)]
        per_cluster[cid] = {
            "n_tables": len(col_sets),
            "avg_jaccard_col_similarity": round(float(np.mean(sims)), 4),
        }

    valid = [v["avg_jaccard_col_similarity"] for v in per_cluster.values()
             if v.get("avg_jaccard_col_similarity") is not None]
    return {
        "per_cluster": per_cluster,
        "mean_intra_cluster_jaccard": round(float(np.mean(valid)), 4) if valid else None,
    }


def health_metrics(all_cluster_sizes, noise_count, n_tables):
    """
    Cluster health indicators.
    """
    non_noise = [s for cid, s in all_cluster_sizes.items() if str(cid) != "-1"]
    if not non_noise:
        return {"note": "no valid clusters"}

    singleton_rate = sum(1 for s in non_noise if s < 2) / len(non_noise)
    doubleton_rate = sum(1 for s in non_noise if s < 3) / len(non_noise)

    return {
        "n_tables_total": n_tables,
        "n_clusters": len(non_noise),
        "noise_count": noise_count,
        "noise_rate": round(noise_count / n_tables, 4) if n_tables else None,
        "cluster_size_min":  int(np.min(non_noise)),
        "cluster_size_max":  int(np.max(non_noise)),
        "cluster_size_mean": round(float(np.mean(non_noise)), 2),
        "cluster_size_std":  round(float(np.std(non_noise)), 2),
        "singleton_rate":    round(singleton_rate, 4),   # clusters with 1 table
        "doubleton_rate":    round(doubleton_rate, 4),   # clusters with < 3 tables
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading embeddings from:      {args.embeddings_path}")
    print(f"Loading cluster summary from: {args.clusters_summary_path}")

    # Load embeddings — keyed by table_id
    raw_embeddings = load_json(args.embeddings_path)
    emb_map = {t["table_id"]: np.array(t["embedding"]) for t in raw_embeddings}
    col_map = {t["table_id"]: set(c.lower() for c in t.get("columns", []))
               for t in raw_embeddings}

    # Load clustering summary
    summary = load_json(args.clusters_summary_path)
    cluster_list = summary["clusters"]   # [{topic_id, num_tables, table_ids}]
    raw_metrics  = summary.get("metrics", {})

    # Build per-cluster embedding and column maps (exclude noise -1)
    cluster_embeddings_map = {}
    cluster_columns_map    = {}
    for c in cluster_list:
        cid = str(c["topic_id"])
        if cid == "-1":
            continue
        embs = [emb_map[tid] for tid in c["table_ids"] if tid in emb_map]
        cols = [col_map[tid] for tid in c["table_ids"] if tid in col_map]
        cluster_embeddings_map[cid] = embs
        cluster_columns_map[cid]    = cols

    # Build flat arrays for sklearn metrics
    all_ids    = []
    all_labels = []
    for c in cluster_list:
        cid = int(c["topic_id"])
        for tid in c["table_ids"]:
            if tid in emb_map:
                all_ids.append(tid)
                all_labels.append(cid)

    embeddings_matrix = np.array([emb_map[tid] for tid in all_ids])
    labels_array      = np.array(all_labels)

    # Compute all metric groups
    print("\nComputing geometric metrics...")
    geo = geometric_metrics(embeddings_matrix, labels_array)

    print("Computing intra-cluster semantic coherence...")
    intra = intra_cluster_coherence(cluster_embeddings_map)

    print("Computing inter-cluster centroid separation...")
    inter = inter_cluster_separation(cluster_embeddings_map)

    print("Computing schema overlap (Jaccard)...")
    schema = schema_overlap_metrics(cluster_columns_map)

    print("Computing health metrics...")
    noise_count = raw_metrics.get("n_noise", 0)
    n_tables    = raw_metrics.get("n_tables", len(all_ids))
    cluster_sizes = raw_metrics.get("cluster_sizes", {})
    health = health_metrics(cluster_sizes, noise_count, n_tables)

    # Assemble full report
    report = {
        "source": {
            "embeddings": args.embeddings_path,
            "clusters_summary": args.clusters_summary_path,
        },
        "geometric_metrics": geo,
        "semantic_coherence": {
            "intra_cluster": {
                "mean_intra_cluster_cosine_sim": intra["mean_intra_cluster_cosine_sim"],
                "per_cluster": intra["per_cluster"],
            },
            "inter_cluster": inter,
        },
        "schema_overlap": schema,
        "cluster_health": health,
    }

    # Print summary to stdout
    sep = "=" * 62
    print(f"\n{sep}")
    print("CLUSTER QUALITY REPORT")
    print(sep)

    print("\n📐 Geometric Metrics (raw embedding space)")
    for k, v in geo.items():
        if isinstance(v, (int, float)):
            interpretation = ""
            if k == "silhouette_score":
                interpretation = "  ← higher better (-1 to 1)"
            elif k == "davies_bouldin_index":
                interpretation = "  ← lower better (≥ 0)"
            elif k == "calinski_harabasz_index":
                interpretation = "  ← higher better"
            print(f"  {k}: {v}{interpretation}")
        elif v:
            print(f"  {k}: {v}")

    print("\n🔗 Semantic Coherence")
    print(f"  mean intra-cluster cosine sim:      {intra['mean_intra_cluster_cosine_sim']}  ← higher better")
    print(f"  mean inter-cluster centroid dist:   {inter.get('mean_centroid_cosine_distance')}  ← higher better")
    print(f"  min  inter-cluster centroid dist:   {inter.get('min_centroid_cosine_distance')}")

    print("\n📋 Schema Overlap (Jaccard on column names)")
    print(f"  mean intra-cluster Jaccard:  {schema['mean_intra_cluster_jaccard']}  ← higher = shared columns")
    for cid, v in schema["per_cluster"].items():
        j = v.get("avg_jaccard_col_similarity")
        print(f"    cluster {cid:>3}: {v['n_tables']} tables, Jaccard={j}")

    print("\n🏥 Cluster Health")
    for k, v in health.items():
        print(f"  {k}: {v}")

    # Save JSON report
    out_path = os.path.join(args.output_dir, "cluster_quality_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Full report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate cluster quality from pre-computed embeddings and BERTopic output"
    )
    parser.add_argument("--embeddings_path", type=str, required=True,
                        help="Path to embeddings JSON (list of {table_id, embedding, columns, ...})")
    parser.add_argument("--clusters_summary_path", type=str, required=True,
                        help="Path to clusters_summary.json from clustering step")
    parser.add_argument("--output_dir", type=str, default="data/output",
                        help="Directory to write cluster_quality_report.json")
    args = parser.parse_args()
    main(args)
