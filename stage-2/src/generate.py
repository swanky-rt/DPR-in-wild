"""
Step 3: Generate Data Product Requests (DPRs) for each cluster.

Takes filtered clusters and uses an LLM with DSPy ChainOfThought
to generate a high-level DPR that covers all tables in each cluster.

Adapted from DPBench generator.py — uses table descriptions
instead of questions (uninformed/offline setting).
"""

import json
import argparse
import os
import logging
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import dspy
from dotenv import load_dotenv

load_dotenv()

# Logging to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("data/output", exist_ok=True)
log_filename = f"data/output/dpr_llm_{timestamp}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# Rotating perspectives assigned to each variant to enforce diversity
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
    Data Product is defined as a self-contained, reusable, and consumable data asset designed to deliver specific value to its users for data-driven use cases. A data product request is a high-level specification of what data and analysis the user needs.

    You are given a cluster containing multiple tables.
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
    - Use a clear, professional tone suitable for real-world user requests, don't just use "analysis".
    - Strictly follow the given perspective to frame your request differently from other variants.

    Output format:
    Return only the final data product request as a single sentence of plain text.

    Examples:
    1. Gather data on hospital readmission rates for heart failure patients across different regions, and analyze which patient demographics or treatment protocols are most strongly associated with reduced readmission.
    2. Compile data on the highest-grossing films of the past 15 years and analyze how factors such as genre, director, production budget, and release season contribute to box office performance.
    3. Collect data showing changes in undergraduate admission rates at top U.S. public universities over the last decade, and assess how SAT scores, tuition, and diversity metrics influence these trends.
    4. Collect data enabling queries on student and professor performance, covering course satisfaction, grades, demographics, GPA, course difficulty, and teaching effectiveness to evaluate student success and professor impact.
    5. Compile a dataset enabling queries on customer spending habits, including payment methods, consumption patterns, average prices, spending at specific locations, and changes over time to support customer segment insights.
    """

    cluster_info: list[dict] = dspy.InputField(
        desc="Cluster of tables with title, columns, and description."
    )
    perspective: str = dspy.InputField(
        desc="The analytical angle to focus on for this specific DPR variant — use it to frame a meaningfully different request."
    )

    data_product_request: str = dspy.OutputField(
        desc="A single-sentence data product request that captures the information needs or intent across all tables, framed through the given perspective. Must be exactly one sentence with no bullet points or line breaks."
    )


def build_cluster_info(tables):
    """Build the cluster_info list for the DSPy signature."""
    cluster_info = []
    for t in tables:
        ci = {
            "title": t.get("title", ""),
            "columns": t.get("columns", []),
            "description": t.get("description", ""),
        }
        cluster_info.append(ci)
    return cluster_info


def generate_dpr_for_cluster(cluster, variant=1, temperature=0.0, max_retries=5):
    """Generate a DPR for a single cluster using ChainOfThought, with retry on rate limits."""
    dpr_id = cluster["dpr_id"]
    tables = cluster["tables"]
    table_ids = [t["table_id"] for t in tables]

    cluster_info = build_cluster_info(tables)
    variant_id = f"{dpr_id}_v{variant}" if variant > 1 else dpr_id

    # Pick a rotating perspective to enforce diversity across variants
    perspective = VARIANT_PERSPECTIVES[(variant - 1) % len(VARIANT_PERSPECTIVES)]

    logging.info("Cluster %s variant %d (temp=%.2f): %d tables — %s",
                 dpr_id, variant, temperature, len(tables), table_ids)
    logging.info("Perspective: %s", perspective)
    logging.info("LLM input: %s", cluster_info)

    cot = dspy.ChainOfThought(DPRGeneration, temperature=temperature)

    for attempt in range(1, max_retries + 1):
        try:
            llm_output = cot(cluster_info=cluster_info, perspective=perspective)
            time.sleep(20)  # pace requests to stay within Groq TPM limit
            break
        except Exception as e:
            err_str = str(e)
            if "rate_limit_exceeded" in err_str or "RateLimitError" in err_str:
                # Parse wait time from Groq error message
                match = re.search(r"try again in ([\d.]+)s", err_str)
                wait = float(match.group(1)) + 1.0 if match else 15.0
                logging.warning("Rate limit on cluster %s variant %d (attempt %d/%d). Waiting %.1fs...",
                                dpr_id, variant, attempt, max_retries, wait)
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
            raise  # re-raise if not a rate limit error or retries exhausted

    dpr_text = llm_output.data_product_request
    reasoning = getattr(llm_output, "reasoning", None)

    logging.info("DPR %s: %s", variant_id, dpr_text)
    if reasoning:
        logging.info("Reasoning: %s", reasoning)

    return {
        "dpr_id": variant_id,
        "cluster_id": dpr_id,
        "variant": variant,
        "temperature": temperature,
        "DPR": dpr_text,
        "reasoning": reasoning,
        "ground_truth": {
            "table_uids": table_ids,
        },
    }


def setup_llm(model, api_base=None, api_key=None):
    """Configure DSPy with the LLM provider via LiteLLM."""
    # DSPy uses LiteLLM under the hood for model routing
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


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load filtered clusters
    with open(args.clusters_path, "r") as f:
        clusters = json.load(f)

    print(f"Loaded {len(clusters)} clusters")

    # Configure LLM
    model = args.model or os.getenv("LLM_MODEL", "openai/gpt-4o")
    setup_llm(model, api_base=args.api_base, api_key=args.api_key)
    print(f"Using model: {model}")

    n_variants = args.n_variants
    temperature = args.temperature

    total_calls = len(clusters) * n_variants
    print(f"Generating {n_variants} variant(s) per cluster = {total_calls} total DPRs")
    print(f"Temperature: {temperature}")

    # Generate DPRs in parallel
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_info = {}
        for cluster in clusters:
            for v in range(1, n_variants + 1):
                future = executor.submit(
                    generate_dpr_for_cluster, cluster, variant=v, temperature=temperature
                )
                future_to_info[future] = (cluster["dpr_id"], v)

        for future in tqdm(
            as_completed(future_to_info),
            total=len(future_to_info),
            desc="Generating DPRs",
        ):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                dpr_id, v = future_to_info[future]
                logging.error("Error on cluster %s variant %d: %s", dpr_id, v, e)
                print(f"Error on cluster {dpr_id} variant {v}: {e}")

    elapsed = time.time() - start_time

    # Sort by cluster_id then variant for consistent output
    results.sort(key=lambda r: (int(r["cluster_id"]), r["variant"]))

    # Compute summary metrics
    n_clusters = len(clusters)
    n_generated = len(results)
    n_failed = n_clusters - n_generated
    total_tables = sum(len(r["ground_truth"]["table_uids"]) for r in results)
    avg_tables_per_dpr = total_tables / n_generated if n_generated else 0
    dpr_lengths = [len(r["DPR"].split()) for r in results]
    avg_dpr_words = sum(dpr_lengths) / n_generated if n_generated else 0

    print(f"\n{'='*60}")
    print("DPR GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Clusters input:       {n_clusters}")
    print(f"DPRs generated:       {n_generated}")
    print(f"Failed:               {n_failed}")
    print(f"Total tables covered: {total_tables}")
    print(f"Avg tables per DPR:   {avg_tables_per_dpr:.1f}")
    print(f"Avg DPR length:       {avg_dpr_words:.0f} words")
    print(f"Time:                 {elapsed:.1f}s")
    print(f"Model:                {model}")

    for r in results:
        print(f"\n  DPR {r['dpr_id']} ({len(r['ground_truth']['table_uids'])} tables, "
              f"{len(r['DPR'].split())} words):")
        print(f"    {r['DPR'][:150]}...")

    # Save as JSONL (one JSON object per line, like DPBench)
    model_short = model.split("/")[-1] if "/" in model else model
    output_path = os.path.join(args.output_dir, f"dprs-{model_short}.jsonl")
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nSaved to {output_path}")
    print(f"Log: {log_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPRs using LLM with ChainOfThought")
    parser.add_argument("--clusters_path", type=str, required=True,
                        help="Path to filtered_clusters.json")
    parser.add_argument("--output_dir", type=str, default="data/output")
    parser.add_argument("--model", type=str, default=None,
                        help="LiteLLM model name (or set LLM_MODEL env var)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="API base URL (or set LLM_API_BASE env var)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (or set OPENAI_API_KEY / GROQ_API_KEY env var)")
    parser.add_argument("--n_variants", type=int, default=3,
                        help="Number of DPR variants per cluster")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="LLM temperature for generation")
    parser.add_argument("--max_workers", type=int, default=2,
                        help="Max parallel LLM calls (keep low for free-tier Groq TPM limits)")
    args = parser.parse_args()
    main(args)
