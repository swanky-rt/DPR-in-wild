"""
Main pipeline: Cluster → Filter → Generate DPRs

Runs all three steps end-to-end.
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Run the full DPR discovery pipeline")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to pre-computed embeddings JSON")
    parser.add_argument("--tables_dir", type=str, default=None,
                        help="Path to tables_clean/ dir for extra metadata")
    parser.add_argument("--output_dir", type=str, default="data/output")

    # Clustering params
    parser.add_argument("--umap_n_neighbors", type=int, default=3)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_n_components", type=int, default=5)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=2)
    parser.add_argument("--hdbscan_epsilon", type=float, default=0.0)
    parser.add_argument("--min_topic_size", type=int, default=2)

    # Filter params
    parser.add_argument("--min_tables", type=int, default=2)
    parser.add_argument("--max_tables", type=int, default=30)

    # Generation params
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--n_variants", type=int, default=3)

    parser.add_argument("--skip_generate", action="store_true",
                        help="Skip DPR generation (run clustering + filtering only)")

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

    # Step 3: Generate DPRs
    if args.skip_generate:
        print("\nSkipping DPR generation (--skip_generate)")
    else:
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

        run_step(os.path.join(src_dir, "generate.py"), gen_args, "DPR Generation")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
