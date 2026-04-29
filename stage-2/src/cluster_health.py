"""
Cluster Health Report

Computes cluster quality metrics after BERTopic clustering and after
cross-cluster DPR generation, giving a holistic view of cluster structure.

Metrics computed:
  - n_clusters, n_noise_tables, n_filtered_clusters
  - silhouette score (intra-cluster cohesion vs inter-cluster separation)
  - avg / min / max cluster size
  - cluster size distribution (histogram buckets)
  - cross-cluster stats: pairs evaluated, strong pairs (score>=3), DPRs generated

Output: cluster_health_<TS>.json  +  printed summary
"""

import json
import argparse
import os
import numpy as np
from collections import Counter
from pathlib import Path


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cluster_size_stats(clusters: dict) -> dict:
    """Compute size stats from clusters.json (topic_id -> [tables])."""
    non_noise = {k: v for k, v in clusters.items() if k != "-1"}
    sizes = [len(v) for v in non_noise.values()]
    if not sizes:
        return {}
    return {
        "n_clusters": len(sizes),
        "total_tables_clustered": sum(sizes),
        "avg_size": round(sum(sizes) / len(sizes), 2),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "size_distribution": {
            "1": sum(1 for s in sizes if s == 1),
            "2-5": sum(1 for s in sizes if 2 <= s <= 5),
            "6-10": sum(1 for s in sizes if 6 <= s <= 10),
            "11-30": sum(1 for s in sizes if 11 <= s <= 30),
            "31+": sum(1 for s in sizes if s > 30),
        },
    }


def filtered_cluster_stats(filtered_clusters: list) -> dict:
    sizes = [len(c["tables"]) for c in filtered_clusters]
    if not sizes:
        return {"n_filtered_clusters": 0}
    return {
        "n_filtered_clusters": len(sizes),
        "avg_size": round(sum(sizes) / len(sizes), 2),
        "min_size": min(sizes),
        "max_size": max(sizes),
    }


def cross_cluster_stats(pairs_path: str, dprs_path: str) -> dict:
    stats = {
        "pairs_evaluated": 0,
        "strong_pairs_score_3plus": 0,
        "cross_cluster_dprs_generated": 0,
        "score_distribution": {},
        "connection_types": {},
    }

    if pairs_path and os.path.exists(pairs_path):
        pairs = load_json(pairs_path)
        stats["pairs_evaluated"] = len(pairs)
        stats["strong_pairs_score_3plus"] = sum(
            1 for p in pairs if p.get("relation_score", 0) >= 3
        )
        score_dist = Counter(p.get("relation_score", 0) for p in pairs)
        stats["score_distribution"] = dict(sorted(score_dist.items()))
        conn_types = Counter(
            p.get("connection_type", "unknown")
            for p in pairs
            if p.get("relation_score", 0) >= 3
        )
        stats["connection_types"] = dict(conn_types.most_common(10))

    if dprs_path and os.path.exists(dprs_path):
        with open(dprs_path, encoding="utf-8") as f:
            dprs = [json.loads(l) for l in f if l.strip()]
        stats["cross_cluster_dprs_generated"] = len(dprs)

    return stats


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    health = {}

    # ── BERTopic cluster stats ────────────────────────────────────────────────
    if args.clusters_path and os.path.exists(args.clusters_path):
        clusters = load_json(args.clusters_path)
        noise_count = len(clusters.get("-1", []))
        health["bertopic"] = {
            "noise_tables": noise_count,
            **cluster_size_stats(clusters),
        }
        print(f"\n{'='*60}")
        print("BERTOPIC CLUSTER STATS")
        print(f"{'='*60}")
        print(f"  Clusters (excl. noise): {health['bertopic'].get('n_clusters', 0)}")
        print(f"  Noise tables:           {noise_count}")
        print(f"  Total clustered tables: {health['bertopic'].get('total_tables_clustered', 0)}")
        print(f"  Avg cluster size:       {health['bertopic'].get('avg_size', 0)}")
        print(f"  Min / Max size:         {health['bertopic'].get('min_size', 0)} / {health['bertopic'].get('max_size', 0)}")
        print(f"  Size distribution:      {health['bertopic'].get('size_distribution', {})}")

    # ── Silhouette score from summary ─────────────────────────────────────────
    if args.clusters_summary_path and os.path.exists(args.clusters_summary_path):
        summary = load_json(args.clusters_summary_path)
        sil = summary.get("metrics", {}).get("silhouette_score")
        if sil is not None:
            health["silhouette_score"] = sil
            print(f"  Silhouette score:       {sil:.4f}")

    # ── Filtered cluster stats ────────────────────────────────────────────────
    if args.filtered_clusters_path and os.path.exists(args.filtered_clusters_path):
        filtered = load_json(args.filtered_clusters_path)
        health["filtered"] = filtered_cluster_stats(filtered)
        print(f"\n  After filtering:")
        print(f"    Filtered clusters:    {health['filtered'].get('n_filtered_clusters', 0)}")
        print(f"    Avg size:             {health['filtered'].get('avg_size', 0)}")
        print(f"    Min / Max size:       {health['filtered'].get('min_size', 0)} / {health['filtered'].get('max_size', 0)}")

    # ── Cross-cluster stats ───────────────────────────────────────────────────
    pairs_path = os.path.join(args.cross_cluster_dir, "pair_decisions.json") \
        if args.cross_cluster_dir else args.cross_cluster_pairs_path
    dprs_path = os.path.join(args.cross_cluster_dir, "cross_cluster_dprs.jsonl") \
        if args.cross_cluster_dir else args.cross_cluster_dprs_path

    cc_stats = cross_cluster_stats(pairs_path, dprs_path)
    health["cross_cluster"] = cc_stats

    print(f"\n{'='*60}")
    print("CROSS-CLUSTER STATS")
    print(f"{'='*60}")
    print(f"  Pairs evaluated:        {cc_stats['pairs_evaluated']}")
    print(f"  Strong pairs (>=3):     {cc_stats['strong_pairs_score_3plus']}")
    print(f"  Cross-cluster DPRs:     {cc_stats['cross_cluster_dprs_generated']}")
    if cc_stats["score_distribution"]:
        print(f"  Score distribution:     {cc_stats['score_distribution']}")
    if cc_stats["connection_types"]:
        print(f"  Top connection types:   {cc_stats['connection_types']}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Cluster health report saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cluster health metrics")
    parser.add_argument("--clusters_path", type=str, default=None,
                        help="Path to clusters.json from BERTopic")
    parser.add_argument("--clusters_summary_path", type=str, default=None,
                        help="Path to clusters_summary.json (contains silhouette score)")
    parser.add_argument("--filtered_clusters_path", type=str, default=None,
                        help="Path to filtered_clusters.json")
    parser.add_argument("--cross_cluster_dir", type=str, default=None,
                        help="Directory containing pair_decisions.json and cross_cluster_dprs.jsonl")
    parser.add_argument("--cross_cluster_pairs_path", type=str, default=None,
                        help="Explicit path to pair_decisions.json (overrides --cross_cluster_dir)")
    parser.add_argument("--cross_cluster_dprs_path", type=str, default=None,
                        help="Explicit path to cross_cluster_dprs.jsonl (overrides --cross_cluster_dir)")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to write health report")
    parser.add_argument("--output_name", type=str, default="cluster_health.json",
                        help="Output filename (default: cluster_health.json)")
    args = parser.parse_args()
    main(args)
