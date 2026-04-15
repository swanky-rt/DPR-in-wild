"""
Generate DPRs for query-matched clusters.
Takes query results with matched clusters and generates 3 DPRs per cluster per query.
"""

import json
import argparse
import os
import logging
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import dspy
from dotenv import load_dotenv

load_dotenv()

# Default paths for online DPR generation
SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2_DIR = SCRIPT_DIR.parent
REPO_ROOT = STAGE2_DIR.parent

DEFAULT_QUERY_RESULTS_PATH = REPO_ROOT / "stage-1" / "query_table_cluster_matches.json"
DEFAULT_TABLES_CLEAN_DIR = REPO_ROOT / "stage-1" / "tables_clean"
DEFAULT_OUTPUT_DIR = STAGE2_DIR / "data" / "output-online"

# Logging setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_filename = DEFAULT_OUTPUT_DIR / f"dpr_queries_{timestamp}.log"

logging.basicConfig(
    filename=str(log_filename),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

VARIANT_PERSPECTIVES = [
    "Focus on relationships and comparisons across the data — what can be contrasted or correlated?",
    "Focus on trends and changes over time — what patterns or progressions does the data reveal?",
    "Focus on decision-making and actionability — what decisions or strategies could this data support?",
    "Focus on specific measurable metrics and quantifiable outcomes — what numbers and statistics matter most?",
    "Focus on a narrative or story angle — what human, geographical, or historical context does the data tell?",
]

class DPRGeneration(dspy.Signature):
    """
    You are a Data Product Request Generator.
    
    # Context:
    Data Product is defined as a self-contained, reusable, and consumable data asset designed to deliver specific value to its users for data-driven use cases.
    
    You are given a cluster containing multiple tables relevant to a user query.
    Each table includes:
    - a title (short description of the table)
    - a list of column headers
    - a natural language description of the table's content
    
    Task:
    Write one data product request that effectively represents the combined data needs across all given tables (data product cluster).
    You will also be given a specific analytical perspective to guide the framing of your request — use it to ensure this variant is meaningfully different from others.
    
    Instructions:
    - The request should incorporate the tables' information in the cluster.
    - Write exactly ONE sentence. Do not use bullet points, numbered lists, or line breaks.
    - Use a clear, professional tone suitable for real-world user requests.
    - Strictly follow the given perspective to frame your request differently from other variants.
    
    Output format:
    Return only the final data product request as a single sentence of plain text.
    """

    cluster_info: list[dict] = dspy.InputField(
        desc="Cluster of tables with title, columns, and description."
    )
    perspective: str = dspy.InputField(
        desc="The analytical angle to focus on for this specific DPR variant."
    )
    user_query: str = dspy.InputField(
        desc="The original user query that motivated this cluster."
    )

    data_product_request: str = dspy.OutputField(
        desc="A single-sentence data product request aligned with the user query."
    )


def build_cluster_info_from_tables(tables_metadata, table_ids):
    """Build cluster_info from table metadata and IDs."""
    cluster_info = []
    for tid in table_ids:
        if tid in tables_metadata:
            t = tables_metadata[tid]
            ci = {
                "title": t.get("title", ""),
                "columns": t.get("columns", []),
                "description": t.get("description", ""),
            }
            cluster_info.append(ci)
    return cluster_info


def generate_dpr_for_query_cluster(query_id, query_text, cluster_id, cluster_tables,
                                   tables_metadata, variant=1, temperature=0.0, max_retries=5):
    """Generate a DPR for a query-matched cluster."""
    
    cluster_info = build_cluster_info_from_tables(tables_metadata, cluster_tables)
    if not cluster_info:
        logging.warning("No table metadata found for cluster %s query %s", cluster_id, query_id)
        return None
    
    variant_id = f"{query_id}_C{cluster_id}_v{variant}"
    perspective = VARIANT_PERSPECTIVES[(variant - 1) % len(VARIANT_PERSPECTIVES)]
    
    logging.info("Query %s, Cluster %s, variant %d: %d tables",
                 query_id, cluster_id, variant, len(cluster_tables))
    logging.info("Perspective: %s", perspective)

    cot = dspy.ChainOfThought(DPRGeneration, temperature=temperature)

    llm_output = None
    for attempt in range(1, max_retries + 1):
        try:
            llm_output = cot(
                cluster_info=cluster_info,
                perspective=perspective,
                user_query=query_text
            )
            time.sleep(20)  # pace requests
            break
        except Exception as e:
            err_str = str(e)
            if "rate_limit_exceeded" in err_str or "RateLimitError" in err_str:
                match = re.search(r"try again in ([\d.]+)s", err_str)
                wait = float(match.group(1)) + 1.0 if match else 15.0
                logging.warning("Rate limit on Q%s C%s v%d (attempt %d/%d). Waiting %.1fs...",
                               query_id, cluster_id, variant, attempt, max_retries, wait)
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
            raise

    if llm_output is None:
        return None

    dpr_text = llm_output.data_product_request
    reasoning = getattr(llm_output, "reasoning", None)

    logging.info("DPR %s: %s", variant_id, dpr_text)

    return {
        "query_id": query_id,
        "query_text": query_text,
        "cluster_id": cluster_id,
        "dpr_id": variant_id,
        "variant": variant,
        "temperature": temperature,
        "DPR": dpr_text,
        "reasoning": reasoning,
        "ground_truth": {
            "table_uids": cluster_tables,
        },
        "num_tables": len(cluster_tables),
    }


def setup_llm(model, api_base=None, api_key=None):
    """Configure DSPy with LLM."""
    api_base = api_base or os.getenv("LLM_API_BASE")
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY", "")

    kwargs = {
        "model": model,
        "max_tokens": 4096,
        "cache": True,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    lm = dspy.LM(**kwargs)
    dspy.configure(lm=lm)


def load_table_metadata(tables_clean_dir):
    """Load table metadata from tables_clean directory."""
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


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load query results with matched clusters chosen earlier
    with open(args.query_results_path, encoding="utf-8") as f:
        query_results = json.load(f)

    # Load table metadata
    tables_metadata = load_table_metadata(args.tables_clean_dir)
    print(f"Loaded metadata for {len(tables_metadata)} tables")

    # Configure LLM
    model = args.model or os.getenv("LLM_MODEL", "openai/gpt-4o")
    setup_llm(model, api_base=args.api_base, api_key=args.api_key)
    print(f"Using model: {model}")

    n_variants = args.n_variants
    temperature = args.temperature

    # Collect all generation tasks
    all_results = []
    total_tasks = 0
    
    print(f"\n{'='*60}")
    print("EXTRACTING GENERATION TASKS FROM QUERIES")
    print(f"{'='*60}")

    for query in query_results.get("query_results", []):
        query_id = query["query_id"]
        query_text = query["query_text"]
        matched_clusters = query.get("matched_clusters", [])
        
        print(f"\nQuery {query_id}: {len(matched_clusters)} matched clusters")
        
        for cluster in matched_clusters:
            cluster_id = cluster["cluster_id"]
            cluster_tables = cluster.get("all_tables_in_cluster", [])
            
            total_tasks += n_variants
            print(f"  Cluster {cluster_id}: {len(cluster_tables)} tables → {n_variants} variants")

    print(f"\nTotal generation tasks: {total_tasks}")
    print(f"Temperature: {temperature}")

    # Generate DPRs in parallel
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_info = {}
        
        for query in query_results.get("query_results", []):
            query_id = query["query_id"]
            query_text = query["query_text"]
            matched_clusters = query.get("matched_clusters", [])
            
            for cluster in matched_clusters:
                cluster_id = cluster["cluster_id"]
                cluster_tables = cluster.get("all_tables_in_cluster", [])
                
                for v in range(1, n_variants + 1):
                    future = executor.submit(
                        generate_dpr_for_query_cluster,
                        query_id, query_text, cluster_id, cluster_tables,
                        tables_metadata, variant=v, temperature=temperature
                    )
                    future_to_info[future] = (query_id, cluster_id, v)

        for future in tqdm(
            as_completed(future_to_info),
            total=len(future_to_info),
            desc="Generating Query-Cluster DPRs",
        ):
            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                query_id, cluster_id, v = future_to_info[future]
                logging.error("Error on Q%s C%s v%d: %s", query_id, cluster_id, v, e)
                print(f"Error on Q{query_id} C{cluster_id} v{v}: {e}")

    elapsed = time.time() - start_time

    # Sort results
    all_results.sort(key=lambda r: (r["query_id"], r["cluster_id"], r["variant"]))

    # Compute metrics
    n_queries = len(query_results.get("query_results", []))
    n_generated = len(all_results)
    total_tables = sum(r["num_tables"] for r in all_results)
    avg_tables = total_tables / n_generated if n_generated else 0
    dpr_lengths = [len(r["DPR"].split()) for r in all_results]
    avg_dpr_words = sum(dpr_lengths) / n_generated if n_generated else 0

    print(f"\n{'='*60}")
    print("DPR GENERATION SUMMARY (QUERY-BASED)")
    print(f"{'='*60}")
    print(f"Total queries:        {n_queries}")
    print(f"DPRs generated:       {n_generated}")
    print(f"Total tables covered: {total_tables}")
    print(f"Avg tables per DPR:   {avg_tables:.1f}")
    print(f"Avg DPR length:       {avg_dpr_words:.0f} words")
    print(f"Time:                 {elapsed:.1f}s")
    print(f"Model:                {model}")

    # Print sample DPRs by query
    for query_id in sorted(set(r["query_id"] for r in all_results)):
        query_dprs = [r for r in all_results if r["query_id"] == query_id]
        print(f"\n  Query {query_id}: {len(query_dprs)} DPRs")
        for r in query_dprs[:2]:
            print(f"    [{r['dpr_id']}] {r['DPR'][:120]}...")

    # Save results as JSONL
    model_short = model.split("/")[-1] if "/" in model else model
    output_path = os.path.join(args.output_dir, f"query_dprs-{model_short}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n\nSaved to {output_path}")
    print(f"Log: {log_filename}")

    # Also save structured JSON for easy query lookup
    by_query = {}
    for result in all_results:
        qid = result["query_id"]
        if qid not in by_query:
            by_query[qid] = []
        by_query[qid].append(result)
    
    structured_path = os.path.join(args.output_dir, f"query_dprs-{model_short}-structured.json")
    with open(structured_path, "w", encoding="utf-8") as f:
        json.dump(by_query, f, indent=2, ensure_ascii=False)
    
    print(f"Structured output: {structured_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DPRs for query-matched clusters"
    )
    parser.add_argument(
        "--query_results_path", type=str, default=str(DEFAULT_QUERY_RESULTS_PATH),
        help="Path to query results JSON with matched clusters"
    )
    parser.add_argument(
        "--tables_clean_dir", type=str, default=str(DEFAULT_TABLES_CLEAN_DIR),
        help="Path to tables_clean directory with table metadata"
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for DPRs"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LiteLLM model name (or set LLM_MODEL env var)"
    )
    parser.add_argument(
        "--api_base", type=str, default=None,
        help="API base URL (or set LLM_API_BASE env var)"
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="API key (or set OPENAI_API_KEY / GROQ_API_KEY env var)"
    )
    parser.add_argument(
        "--n_variants", type=int, default=3,
        help="Number of DPR variants per cluster per query"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="LLM temperature for generation"
    )
    parser.add_argument(
        "--max_workers", type=int, default=2,
        help="Max parallel LLM calls"
    )
    
    args = parser.parse_args()
    main(args)