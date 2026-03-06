# Stage 3: SQL Generation & Grounding — Implementation Summary & Presentation Guide

This document explains **how Stage 3 is implemented** and **how to present it** clearly to an audience (e.g., stakeholders, reviewers, or a thesis defense).

---

## 1. High-Level: What Stage 3 Does

**Input:** Stage 2 output — a list of **Data Product Requests (DPRs)**. Each DPR is a natural-language request (e.g., *"Compile a comprehensive dataset of Weird Al Yankovic's filmography..."*) plus **ground-truth table IDs** (e.g., T1, T2, T3).

**Output:** For each DPR, Stage 3 produces:
- **Decomposed sub-questions** (atomic, SQL-answerable)
- **Per–sub-question SQL** with execution results (row counts, previews)
- **Grounded mini-summaries** (facts derived only from query results)
- A **final grounded summary** that answers the DPR and is ready for Stage 4 (LLM-as-a-Judge)

**Key idea:** We moved from a **single-SQL-per-DPR** approach to an **agentic discovery loop**: decompose → generate SQL → execute → self-correct on errors or empty results → summarize. This improves grounding and reduces LLM hallucination (invented columns, bad joins, wrong filter values).

---

## 2. End-to-End Flow (One Slide / Diagram)

```
Stage 2 (DPRs + table_uids)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: DPR DECOMPOSITION                                      │
│  LLM breaks each DPR into 3–5 atomic sub-questions.              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  DATABASE SETUP (per DPR)                                        │
│  • Load table metadata from T1..T10 JSON files (folder or file) │
│  • Build in-memory SQLite DB: schema + sample rows from JSON     │
│  • Schema: SQL-safe column names, types (TEXT/REAL), no Time→REAL│
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: AGENTIC LOOP (per sub-question)                        │
│  • Fetch sample rows from DB → show LLM real values              │
│  • Generate SQL (schema + samples in prompt)                      │
│  • Reject cartesian joins (ON 1=1, CROSS JOIN) before execution  │
│  • Execute → if error or 0 rows → refine_sql_with_error (up to 3) │
│  • On success: mini-summary from result preview                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: GROUNDING LAYER                                        │
│  • Collect all mini-summaries                                    │
│  • generate_final_summary(DPR, mini_summaries) → one paragraph  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
Stage 3 output JSON (per DPR: sub_questions, subquery_results, mini_summaries, final_summary, …)
         │
         ▼
Stage 4: LLM-as-a-Judge scores final_summary vs original DPR (0.0–1.0)
```

---

## 3. Implementation Details (For Deep-Dive Slides)

### 3.1 Phase 1: DPR Decomposition

- **Function:** `generate_subquestions(client, model, dpr_text, max_questions=5)`
- **Behavior:** One LLM call. Prompt asks for a **JSON list of strings**: 3–5 concise, non-overlapping, SQL-answerable sub-questions implied by the DPR.
- **Fallback:** If parsing fails, we use the full DPR as a single sub-question.
- **Why it matters:** Complex DPRs (e.g., “filmography + evolution + collaborations”) are split into simpler pieces (“List titles”, “List release years”, “List roles”), so each piece gets a focused SQL query instead of one huge, error-prone query.

### 3.2 Database: Schema + Real Data Grounding

- **Function:** `_build_empty_db_from_table_metadata(table_metas)`
- **Input:** Table metadata for the DPR’s tables only (e.g., T1, T2, T3). Metadata comes from `_load_tables_meta(path)`, which supports:
  - A **directory** of JSON files (e.g., `data/stage1/tables_clean/` with T1.json … T10.json)
  - A **single** `tables.json` file (dict of table_id → meta).
- **What we build:**
  - **Schema:** For each table we create SQLite tables with SQL-safe column names (spaces → underscores; reserved words prefixed). Column types: `REAL` only for columns in `numeric_columns` and **not** time-like (e.g., “Time”, “Duration”) to avoid values like `"55:06"` being mis-stored.
  - **Data:** We read `meta["rows"]` from each T*.json. We support:
    - **Row dicts:** `[{ "Year": 1988, "Title": "Tapeheads", "Role": "Himself" }, ...]`
    - **Flattened cells (HybridQA):** `[{ "value": "1988", "urls": [] }, { "value": "Tapeheads", ... }, ...]` grouped by column count.
  - We insert up to `MAX_SAMPLE_ROWS_PER_TABLE` (e.g., 10) rows per table so that **SQL runs against real data**, not empty tables.
- **Result:** Execution returns real `row_count` and `preview`; mini-summaries are grounded in actual cells.

### 3.3 Phase 2: Self-Correcting SQL Loop (Per Sub-Question)

For **each sub-question** we run up to **3 attempts**:

1. **Sample rows for the LLM**  
   `_fetch_table_samples(cursor, table_uids)` runs `SELECT * FROM Tn LIMIT 5` for each table and formats the result as JSON. This is passed into both `generate_sql` and `refine_sql_with_error` so the model sees **real values** (e.g., `Role = 'Himself'`) and is instructed to prefer them over invented ones (e.g., `Role = 'Weird Al Yankovic'`).

2. **Generate SQL**  
   `generate_sql(client, model, question, schema_string, samples_string)`  
   - Prompt includes: schema, **example rows**, question, and strict rules: no invented columns/tables, JOIN only when there is a clear shared column, no `ON 1=1` or `CROSS JOIN`, prefer single-table queries when there’s no obvious join key.

3. **Sanitization before execution**  
   `_is_cartesian_sql(sql)` checks for `ON 1=1`, `ON 1 = 1`, or `CROSS JOIN`. If detected, we **do not execute**; we record a synthetic error and treat it as retryable so the next attempt can produce a valid join or a single-table query.

4. **Execute**  
   `execute_and_validate(cursor, sql, require_non_empty)` runs the SQL. If `--require-non-empty` is set and the result is 0 rows, we return `ok=False` with a specific message so the pipeline treats it as “empty result” and retries.

5. **Decide: success or retry**  
   - **Success:** `ok and row_count > 0` → we keep this SQL and its preview, generate a mini-summary, and move to the next sub-question.
   - **Retry:** We retry if the failure is due to: schema/syntax error (`_is_schema_error`), cartesian join, explicit empty-result error (`_is_empty_result_error`), or “executed but 0 rows”. Otherwise we stop retrying for this sub-question.

6. **Refinement**  
   `refine_sql_with_error(client, model, question, schema_string, previous_sql, error_message, empty_result, samples_string)`  
   - Prompt includes: schema, **same sample rows**, question, previous SQL, and feedback (error text and/or “returned 0 rows, try broader query, e.g. LIKE”).  
   - The model is asked to fix the SQL without inventing columns/tables and to prefer values from the example rows.

So: **schema grounding** (only allowed columns/tables), **data grounding** (sample rows in prompt + real rows in DB), and **join discipline** (no cartesian, optional single-table) together reduce semantic misunderstandings.

### 3.4 Phase 3: Grounding Layer (Mini-Summaries → Final Summary)

- **Per sub-question (on success):** `summarize_subquestion_result(client, model, dpr_text, sub_question, table_uids, preview_rows)`  
  - Input: the DPR, the sub-question, and the **preview rows** (first few result rows).  
  - Output: 1–2 sentences stating only what the preview supports; no SQL or table names.

- **Per DPR:** `generate_final_summary(client, model, dpr_text, mini_summaries)`  
  - Input: the DPR and the list of mini-summaries.  
  - Output: one concise paragraph that answers the DPR using only those findings, in narrative form. This is the **final_summary** that Stage 4 judges.

---

## 4. Output Structure (`stage3_output_final.json`)

Each element in the top-level JSON array is **one DPR**. Representative fields:

| Field | Meaning |
|-------|--------|
| `dpr_id` | Identifier from Stage 2 (e.g. `"1"`, `"1_v2"`) |
| `DPR` | Original natural-language request |
| `sub_questions` | List of 3–5 atomic questions from Phase 1 |
| `subquery_results` | One object per sub-question: `sub_question`, `attempts` (list of `{ sql, execution_status, error, row_count }`), `final_sql`, `final_execution_status`, `final_row_count`, `mini_summary` |
| `mini_summaries` | List of grounded 1–2 sentence facts (one per successful sub-question) |
| `final_summary` | Single paragraph synthesizing all mini-summaries; **input to Stage 4 judge** |
| `tables` | Table IDs used (e.g. `["T1","T2","T3"]`) |
| `ground_truth` | From Stage 2 (e.g. `{ "table_uids": ["T1","T2","T3"] }`) |
| `generated_sql` | Representative SQL (e.g. first successful sub-query) for backward compatibility |
| `execution_status` | Whether at least one sub-query succeeded |
| `result` | Aggregated `validation`, `row_count`, `preview` (or `error`) |
| `schema_mapping` | Original column name → SQL-safe name per table |
| `llm_model` | Model used (e.g. `llama-3.1-8b-instant`) |


---

## 6. One-Paragraph Summary (For Abstract or Intro)

*Stage 3 turns Stage 2 DPRs into grounded, judgeable answers. It decomposes each DPR into atomic sub-questions, builds an in-memory SQLite database from table metadata and sample rows (T1..T10 JSONs), and runs an agentic loop per sub-question: the LLM generates SQL using schema and example rows, the pipeline blocks cartesian joins and executes queries, and on error or empty result it refines the SQL (up to 3 attempts). Successful results are summarized into mini-summaries and then into a single final summary per DPR. This design enforces schema and data grounding and reduces join/schema/data hallucination, so Stage 4 can reliably score how well the final summary reflects the DPR given the actual data.*

---

## 7. File and Config Reference

- **Code:** `src/sql_grounding/pipeline.py`
- **Output:** `stage3_output_final.json` (or path passed via `-o`)
- **Run (example):**  
  `python src/sql_grounding/pipeline.py -i data/stage2/output/run_test_groq/dprs.json -o stage3_output_final.json -n 5 --tables-meta data/stage1/tables_clean --require-non-empty`
- **Constants:** `MAX_SAMPLE_ROWS_PER_TABLE` (rows inserted per table), `LLM_SAMPLE_ROWS_PER_TABLE` (rows shown in prompt per table).
