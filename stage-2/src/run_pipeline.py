"""
Main pipeline: Cluster → Filter → Generate DPRs → Cross-cluster DPRs → Merge
"""

import argparse
import json
import os
import subprocess
import sys


def run_step(script, args_list, step_name):
    """Run a pipeline step as a subprocess."""
    cmd = [sys.executable, script] + args_list
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed with return code {result.returncode}")
        sys.exit(1)


def merge_dprs(main_path, cross_path, output_path):
    """Merge main DPRs and cross-cluster DPRs into a single JSONL."""
    records = []

    with open(main_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n_main = len(records)

    with open(cross_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                # Normalize to common schema
                r.setdefault("variant", None)
                r.setdefault("temperature", None)
                r["source"] = "cross_cluster"
                records.append(r)

    # Tag main DPRs with source
    for r in records[:n_main]:
        r["source"] = "single_cluster"

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return n_main, len(records) - n_main


def main():
    parser = argparse.ArgumentParser(description="Run the full DPR discovery pipeline")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to pre-computed embeddings JSON")
    parser.add_argument("--tables_dir", type=str, default=None,
                        help="Path to tables_clean/ dir for extra metadata")
    parser.add_argument("--output_dir", type=str, default="data/output")

    # Clustering params
    parser.add_argument("--umap_n_neighbors", type=int, default=10)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_n_components", type=int, default=10)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=3)
    parser.add_argument("--hdbscan_epsilon", type=float, default=0.0)
    parser.add_argument("--min_topic_size", type=int, default=3)

    # Filter params
    parser.add_argument("--min_tables", type=int, default=2)
    parser.add_argument("--max_tables", type=int, default=30)

    # Generation params
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--n_variants", type=int, default=3)

    # Cross-cluster params
    parser.add_argument("--cross_cluster_top_k", type=int, default=20,
                        help="Number of most dissimilar cluster pairs to evaluate for cross-cluster DPRs")
    parser.add_argument("--cross_cluster_sleep", type=float, default=5.0)
    parser.add_argument("--skip_cross_cluster", action="store_true",
                        help="Skip cross-cluster DPR generation")

    parser.add_argument("--skip_generate", action="store_true",
                        help="Skip all DPR generation (clustering + filtering only)")

    args = parser.parse_args()

    src_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Cluster
    cluster_args = [
        "--input_path", args.input_path,
        "--output_dir", args.output_dir,
        "--umap_n_neighbors", str(args.umap_n_neighbors),
        "--umap_min_dist", str(args.umap_min_dist),
        "--umap_n_components", str(args.umap_n_components),
        "--hdbscan_min_cluster_size", str(args.hdbscan_min_cluster_size),
        "--hdbscan_epsilon", str(args.hdbscan_epsilon),
        "--min_topic_size", str(args.min_topic_size),
    ]
    if args.tables_dir:
        cluster_args += ["--tables_dir", args.tables_dir]

    run_step(os.path.join(src_dir, "cluster.py"), cluster_args, "Clustering (BERTopic)")

    # Step 2: Filter
    clusters_path = os.path.join(args.output_dir, "clusters.json")
    filter_args = [
        "--clusters_path", clusters_path,
        "--output_dir", args.output_dir,
        "--min_tables", str(args.min_tables),
        "--max_tables", str(args.max_tables),
    ]
    run_step(os.path.join(src_dir, "filter.py"), filter_args, "Filtering")

    if args.skip_generate:
        print("\nSkipping DPR generation (--skip_generate)")
    else:
        # Step 3: Generate single-cluster DPRs
        filtered_path = os.path.join(args.output_dir, "filtered_clusters.json")
        gen_args = [
            "--clusters_path", filtered_path,
            "--output_dir", args.output_dir,
            "--n_variants", str(args.n_variants),
        ]
        if args.model:
            gen_args += ["--model", args.model]
        if args.api_base:
            gen_args += ["--api_base", args.api_base]
        if args.api_key:
            gen_args += ["--api_key", args.api_key]

        run_step(os.path.join(src_dir, "generate.py"), gen_args, "DPR Generation (single-cluster)")

        # Step 4: Cross-cluster DPR generation
        if not args.skip_cross_cluster:
            cross_dir = os.path.join(args.output_dir, "cross_cluster")
            cross_args = [
                "--clusters_path", clusters_path,
                "--embeddings_path", args.input_path,
                "--output_dir", cross_dir,
                "--top_k", str(args.cross_cluster_top_k),
                "--sleep_between", str(args.cross_cluster_sleep),
            ]
            if args.model:
                cross_args += ["--model", args.model]
            if args.api_base:
                cross_args += ["--api_base", args.api_base]
            if args.api_key:
                cross_args += ["--api_key", args.api_key]

            run_step(
                os.path.join(src_dir, "experiments", "cross_cluster", "generate.py"),
                cross_args,
                "Cross-cluster DPR Generation",
            )

            # Merge outputs
            model_short = (args.model or "model").split("/")[-1]
            main_dprs = os.path.join(args.output_dir, f"dprs-{model_short}.jsonl")
            cross_dprs = os.path.join(cross_dir, "cross_cluster_dprs.jsonl")
            merged_path = os.path.join(args.output_dir, f"dprs-{model_short}-merged.jsonl")

            if os.path.exists(main_dprs) and os.path.exists(cross_dprs):
                print(f"\n{'='*60}")
                print("STEP: Merging DPR outputs")
                print(f"{'='*60}")
                n_main, n_cross = merge_dprs(main_dprs, cross_dprs, merged_path)
                print(f"  Single-cluster DPRs: {n_main}")
                print(f"  Cross-cluster DPRs:  {n_cross}")
                print(f"  Total:               {n_main + n_cross}")
                print(f"  Saved to: {merged_path}")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
