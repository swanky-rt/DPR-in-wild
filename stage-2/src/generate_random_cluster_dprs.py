"""
Stage 2c: Query-guided random cluster DPR generation.

For each of the first N user queries:
  - randomly sample K clusters from filtered_clusters.json
  - generate 1 DPR per (query, cluster) pair using the LLM
  - total: N queries × K samples = N*K DPRs

Default: 100 queries × 50 samples = 5,000 DPRs

Output: <output_dir>/<output_name>.jsonl
        <output_dir>/<output_name>-structured.json
"""

import json
import argparse
import os
import logging
import time
import random
import re
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dspy
from dotenv import load_dotenv
from tqdm import tqdm


def start_heartbeat(label, counter, total, interval=30):
    """Print progress every `interval` seconds. Returns a stop event."""
    stop = threading.Event()
    start_time = time.time()

    def _beat():
        while not stop.wait(interval):
            elapsed = int(time.time() - start_time)
            done = counter[0]
            pct = 100 * done / total if total else 0
            print(
                f"  [heartbeat] {label} — {done}/{total} ({pct:.1f}%) "
                f"elapsed {elapsed//60}m{elapsed%60:02d}s",
                flush=True,
            )

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    return stop

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2_DIR = SCRIPT_DIR.parent
REPO_ROOT = STAGE2_DIR.parent

DEFAULT_FILTERED_CLUSTERS_PATH = STAGE2_DIR / "data" / "output" / "filtered_clusters.json"
DEFAULT_TABLES_CLEAN_DIR = REPO_ROOT / "stage-1" / "tables_clean"
DEFAULT_OUTPUT_DIR = STAGE2_DIR / "data" / "output-random"
DEFAULT_QUERIES_FILE = REPO_ROOT / "ward" / "user_queries_only.txt"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_filename = DEFAULT_OUTPUT_DIR / f"random_cluster_dprs_{timestamp}.log"

logging.basicConfig(
    filename=str(log_filename),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ── DSPy signature (same as generate_dprs_for_queries.py) ────────────────────

class DPRGeneration(dspy.Signature):
    """
    You are a Data Product Request Generator.

    Data Product is defined as a self-contained, reusable, and consumable data asset
    designed to deliver specific value to its users for data-driven use cases.

    You are given a cluster containing multiple tables relevant to a user query.
    Each table includes:
    - a title (short description of the table)
    - a list of column headers
    - a natural language description of the table's content

    Task:
    Write one data product request that effectively represents the combined data needs
    across all given tables (data product cluster).

    Instructions:
    - The request should incorporate the tables' information in the cluster.
    - Write exactly ONE sentence. Do not use bullet points, numbered lists, or line breaks.
    - Use a clear, professional tone suitable for real-world user requests.
    - Frame the request in the context of the user query provided.

    Output format:
    Return only the final data product request as a single sentence of plain text.
    """

    cluster_info: list[dict] = dspy.InputField(
        desc="Cluster of tables with title, columns, and description."
    )
    user_query: str = dspy.InputField(
        desc="The user query that motivates this DPR."
    )

    data_product_request: str = dspy.OutputField(
        desc="A single-sentence data product request aligned with the user query."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_queries(queries_file: str, n_queries: int):
    """Load first n_queries from a .txt (one per line) or JSON file."""
    path = str(queries_file)
    queries = []

    if path.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    queries.append({"query_id": f"Q{i+1}", "query_text": line})
    else:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for i, entry in enumerate(raw):
            query_id = entry.get("query_id") or entry.get("dpr_id") or f"Q{i+1}"
            query_text = entry.get("query_text") or entry.get("user_query") or ""
            if query_text:
                queries.append({"query_id": query_id, "query_text": query_text})

    return queries[:n_queries]


def load_filtered_clusters(path: str):
    """Load filtered_clusters.json → list of {dpr_id, cluster_key, tables}."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_table_metadata(tables_clean_dir: str):
    """Load all table metadata from tables_clean/ directory."""
    metadata = {}
    if not os.path.isdir(tables_clean_dir):
        return metadata
    for fname in os.listdir(tables_clean_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(tables_clean_dir, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    table = json.load(f)
                tid = table.get("table_id", fname.replace(".json", ""))
                metadata[tid] = table
            except Exception as e:
                logging.warning("Failed to load %s: %s", fname, e)
    return metadata


def build_cluster_info(cluster, tables_metadata):
    """Build cluster_info list from a cluster's tables."""
    cluster_info = []
    for t in cluster.get("tables", []):
        tid = t.get("table_id", "")
        meta = tables_metadata.get(tid, t)
        cluster_info.append({
            "title": meta.get("title", t.get("title", "")),
            "columns": meta.get("columns", t.get("columns", [])),
            "description": meta.get("description", t.get("description", "")),
        })
    return cluster_info


def setup_llm(model, api_base=None, api_key=None):
    api_base = api_base or os.getenv("LLM_API_BASE")
    api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    kwargs = {"model": model, "max_tokens": 4096, "cache": True}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    dspy.configure(lm=dspy.LM(**kwargs))


def generate_one_dpr(query_id, query_text, cluster, tables_metadata,
                     sample_idx, temperature=1.0, max_retries=5):
    """Generate a single DPR for a (query, cluster) pair."""
    cluster_id = cluster["dpr_id"]
    cluster_key = cluster.get("cluster_key", cluster_id)
    table_ids = [t["table_id"] for t in cluster.get("tables", [])]

    cluster_info = build_cluster_info(cluster, tables_metadata)
    if not cluster_info:
        logging.warning("No table info for cluster %s query %s", cluster_id, query_id)
        return None

    dpr_id = f"{query_id}_C{cluster_id}_s{sample_idx}"
    logging.info("Generating DPR %s: query=%s cluster=%s tables=%d",
                 dpr_id, query_id, cluster_id, len(table_ids))

    cot = dspy.ChainOfThought(DPRGeneration, temperature=temperature)

    llm_output = None
    for attempt in range(1, max_retries + 1):
        try:
            llm_output = cot(cluster_info=cluster_info, user_query=query_text)
            time.sleep(20)
            break
        except Exception as e:
            err_str = str(e)
            if "rate_limit_exceeded" in err_str or "RateLimitError" in err_str:
                match = re.search(r"try again in ([\d.]+)s", err_str)
                wait = float(match.group(1)) + 1.0 if match else 15.0
                logging.warning("Rate limit on %s (attempt %d/%d). Waiting %.1fs...",
                                dpr_id, attempt, max_retries, wait)
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
            raise

    if llm_output is None:
        return None

    return {
        "dpr_id": dpr_id,
        "query_id": query_id,
        "query_text": query_text,
        "cluster_id": cluster_id,
        "cluster_key": cluster_key,
        "sample_idx": sample_idx,
        "DPR": llm_output.data_product_request,
        "reasoning": getattr(llm_output, "reasoning", None),
        "ground_truth": {"table_uids": table_ids},
        "num_tables": len(table_ids),
        "temperature": temperature,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load inputs
    queries = load_queries(args.queries_file, args.n_queries)
    print(f"Loaded {len(queries)} queries (capped at {args.n_queries})")

    clusters = load_filtered_clusters(args.filtered_clusters_path)
    print(f"Loaded {len(clusters)} filtered clusters")

    tables_metadata = load_table_metadata(args.tables_clean_dir)
    print(f"Loaded metadata for {len(tables_metadata)} tables")

    # Setup LLM
    model = args.model or os.getenv("LLM_MODEL")
    if not model:
        raise ValueError("LLM model not specified. Set LLM_MODEL env var or pass --model.")
    setup_llm(model, api_base=args.api_base, api_key=args.api_key)
    print(f"Using model: {model}")

    n_clusters = len(clusters)
    n_samples = args.n_samples_per_query
    total_tasks = len(queries) * n_samples

    print(f"\n{'='*60}")
    print(f"RANDOM CLUSTER DPR GENERATION")
    print(f"{'='*60}")
    print(f"Queries:            {len(queries)}")
    print(f"Clusters available: {n_clusters}")
    print(f"Samples per query:  {n_samples}")
    print(f"Total DPRs:         {total_tasks}")
    print(f"Sampling:           with replacement (duplicates allowed)")

    # Build all (query, cluster, sample_idx) tasks
    # Pick one cluster at a time — duplicates allowed (same cluster can appear
    # multiple times for the same query)
    tasks = []
    random.seed(args.seed)
    for query in queries:
        for idx in range(1, n_samples + 1):
            cluster = random.choice(clusters)
            tasks.append((query, cluster, idx))

    # Generate DPRs in parallel
    all_results = []
    start_time = time.time()
    counter = [0]  # mutable for heartbeat thread

    stop_hb = start_heartbeat("Stage 2c random DPR gen", counter, total_tasks)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {}
        for query, cluster, idx in tasks:
            future = executor.submit(
                generate_one_dpr,
                query["query_id"], query["query_text"],
                cluster, tables_metadata, idx,
                temperature=args.temperature,
            )
            future_to_task[future] = (query["query_id"], cluster["dpr_id"], idx)

        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Generating DPRs",
            unit="dpr",
            dynamic_ncols=True,
        ):
            query_id, cluster_id, idx = future_to_task[future]
            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                logging.error("Error on %s C%s s%d: %s", query_id, cluster_id, idx, e)
                print(f"Error on {query_id} C{cluster_id} s{idx}: {e}")
            finally:
                counter[0] += 1

    stop_hb.set()
    elapsed = time.time() - start_time

    # Sort for deterministic output
    all_results.sort(key=lambda r: (r["query_id"], r["cluster_id"], r["sample_idx"]))

    # Summary
    n_generated = len(all_results)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Requested:    {total_tasks}")
    print(f"Generated:    {n_generated}")
    print(f"Time:         {elapsed:.1f}s")
    print(f"Model:        {model}")

    # Save JSONL (append mode if --append, otherwise overwrite)
    stem = args.output_name or f"random_cluster_dprs"
    if stem.endswith(".jsonl"):
        stem = stem[:-6]

    jsonl_path = os.path.join(args.output_dir, f"{stem}.jsonl")
    mode = "a" if args.append else "w"
    with open(jsonl_path, mode, encoding="utf-8") as f:
        for r in all_results:
            r["source"] = "random_cluster"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    action = "Appended" if args.append else "Saved"
    print(f"\n{action} {n_generated} DPRs → {jsonl_path}")

    # Save structured JSON grouped by query
    by_query = {}
    for r in all_results:
        by_query.setdefault(r["query_id"], []).append(r)

    structured_path = os.path.join(args.output_dir, f"{stem}-structured.json")
    with open(structured_path, "w", encoding="utf-8") as f:
        json.dump(by_query, f, indent=2, ensure_ascii=False)
    print(f"Saved structured → {structured_path}")
    print(f"Log: {log_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DPRs by pairing user queries with randomly sampled clusters"
    )
    parser.add_argument("--filtered_clusters_path", type=str,
                        default=str(DEFAULT_FILTERED_CLUSTERS_PATH),
                        help="Path to filtered_clusters.json from stage 2a")
    parser.add_argument("--tables_clean_dir", type=str,
                        default=str(DEFAULT_TABLES_CLEAN_DIR),
                        help="Path to tables_clean/ directory")
    parser.add_argument("--queries_file", type=str,
                        default=str(DEFAULT_QUERIES_FILE),
                        help="Path to user queries (.txt one per line, or .json)")
    parser.add_argument("--output_dir", type=str,
                        default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Stem for output filenames (default: random_cluster_dprs)")
    parser.add_argument("--n_queries", type=int, default=100,
                        help="Number of queries to use (default: 100)")
    parser.add_argument("--n_samples_per_query", type=int, default=50,
                        help="Number of random cluster samples per query (default: 50)")
    parser.add_argument("--model", type=str, default=None,
                        help="LiteLLM model name")
    parser.add_argument("--api_base", type=str, default=None,
                        help="API base URL")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="LLM temperature (default: 1.0)")
    parser.add_argument("--max_workers", type=int, default=2,
                        help="Max parallel LLM calls (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for cluster sampling (default: 42)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output file instead of overwriting")

    args = parser.parse_args()
    main(args)
