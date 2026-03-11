# Stage 3: SQL Generation & Grounding — Implementation Summary & Presentation Guide

---

## 1. High-Level: What Stage 3 Does

**Input:** 
Stage 2 output — a list of **Data Product Requests (DPRs)**. Each DPR is a natural-language request (e.g., *"Compile a comprehensive dataset of Weird Al Yankovic's filmography..."*) plus **ground-truth table IDs** (e.g., T1, T2, T3).

**Output:** 
For each DPR, Stage 3 produces:
- **Decomposed sub-questions** (atomic, SQL-answerable)
- **Per–sub-question SQL** with execution results (row counts, previews)
- **Grounded mini-summaries** (facts derived only from query results)
- A **final grounded summary** that answers the DPR and is ready for Stage 4 (LLM-as-a-Judge)

---

## 2. End-to-End Flow (One Slide / Diagram)

```
Stage 2 (DPRs + table_uids)
         │
         ▼
  PHASE 1: DPR DECOMPOSITION                                      
  LLM breaks each DPR into 3–5 atomic sub-questions.              

         │
         ▼
  DATABASE SETUP (per DPR)                                        
  Load table metadata from T1..T10 JSON files (folder or file) 
  Build in-memory SQLite DB: schema + sample rows from JSON     
         │
         ▼

  PHASE 2: AGENTIC LOOP (per sub-question)                        
  Fetch sample rows from DB → show LLM real values              
  Generate SQL (schema + samples in prompt)                      
  Execute → if error or 0 rows → refine_sql_with_error (up to 3) 
  On success: mini-summary from result preview                  
         │
         ▼
  PHASE 3: GROUNDING LAYER                                        
  Collect all mini-summaries                                    
  generate_final_summary(DPR, mini_summaries) → one summary  
         │
         ▼
Stage 3 output JSON (per DPR: sub_questions, subquery_results, mini_summaries, final_summary, …)

```
---

## 3. Implementation Details

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
  - **Data:** We read `meta["rows"]` from each T*.json. We support:
    - **Row dicts:** `[{ "Year": 1988, "Title": "Tapeheads", "Role": "Himself" }, ...]`
    - **Flattened cells (HybridQA):** `[{ "value": "1988", "urls": [] }, { "value": "Tapeheads", ... }, ...]` grouped by column count.
  - We insert up to `MAX_SAMPLE_ROWS_PER_TABLE` (e.g., 10) rows per table so that **SQL runs against real data**, not empty tables.
- **Result:** Execution returns real `row_count` and `preview`; mini-summaries are grounded in actual cells.

### 3.3 Phase 2: Self-Correcting SQL Loop (Per Sub-Question)

For **each sub-question** we run up to **3 attempts**:

1. **Sample rows for the LLM**  
   `_fetch_table_samples(cursor, table_uids)` runs `SELECT * FROM Tn LIMIT 5` for each table and formats the result as JSON. This is passed into both `generate_sql` and `refine_sql_with_error` so the model sees **real values** (e.g., `Role = 'Himself'`).

2. **Generate SQL**  
   `generate_sql(client, model, question, schema_string, samples_string)` and Prompt includes: schema, **example rows**, question, and strict rules

   `llm_model`: Model used `llama-3.1-8b-instant`

3. **Sanitization before execution**  
   `_is_cartesian_sql(sql)` checks for `ON 1=1`, `ON 1 = 1`, or `CROSS JOIN`. If detected, we **do not execute**; we record a synthetic error and treat it as retryable so the next attempt can produce a valid join or a single-table query.

4. **Execute**  
   `execute_and_validate(cursor, sql, require_non_empty)` runs the SQL. If `--require-non-empty` is set and the result is 0 rows, we return `ok=False` with a specific message so the pipeline treats it as “empty result” and retries.

5. **success or retry**  
   - **Success:** `ok and row_count > 0` → we keep this SQL and its preview, generate a mini-summary, and move to the next sub-question.
   - **Retry:** We retry if the failure is due to: schema/syntax error (`_is_schema_error`), cartesian join, explicit empty-result error (`_is_empty_result_error`), or “executed but 0 rows”. 

6. **Refinement**  
   `refine_sql_with_error(client, model, question, schema_string, previous_sql, error_message, empty_result, samples_string)` and schema includes **same sample rows**, question, previous SQL, and feedback (error text and/or “returned 0 rows, try broader query).  
   - The model is asked to fix the SQL without inventing columns/tables and to prefer values from the example rows.
   
   **schema grounding** (only allowed columns/tables), **data grounding** (sample rows in prompt + real rows in DB), and **join discipline** (no cartesian, optional single-table) together reduce semantic misunderstandings.

### 3.4 Phase 3: Grounding Layer (Mini-Summaries → Final Summary)

- **Per sub-question (on success):** `summarize_subquestion_result(client, model, dpr_text, sub_question, table_uids, preview_rows)`  
  - Input: the DPR, the sub-question, and the **preview rows** (first few result rows).  
  - Output: 1–2 sentences stating only what the preview supports; no SQL or table names.

- **Per DPR:** `generate_final_summary(client, model, dpr_text, mini_summaries)`  
  - Input: the DPR and the list of mini-summaries.  
  - Output: one concise paragraph that answers the DPR using only those findings, in narrative form. This is the **final_summary** that Stage 4 judges.

---

## 4. Output Structure (`stage3_output_final.json`)

{
    "dpr_id": "1",
    "DPR": "Compile a comprehensive dataset of \"Weird Al\" Yankovic's filmography and media appearances, including his roles in films, television shows, and other productions, and analyze how his involvement in different types of media and his portrayal in various roles have evolved over time, as well as identify any notable patterns or collaborations throughout his career.",
    "sub_questions": [
      "What are the titles of films and television shows that 'Weird Al' Yankovic has been involved in?",
      "What are the different types of media that 'Weird Al' Yankovic has been involved in?",
      "What are the notable collaborations and patterns in 'Weird Al' Yankovic's filmography and media appearances?"
    ],
    "subquery_results": [
      {
        "sub_question": "What are the titles of films and television shows that 'Weird Al' Yankovic has been involved in?",
        "attempts": [
          {
            "sql": "SELECT Title FROM T1 WHERE Role = 'Himself'",
            "execution_status": true,
            "error": null,
            "row_count": 6
          }
        ],
        "final_sql": "SELECT Title FROM T1 WHERE Role = 'Himself'",
        "final_execution_status": true,
        "final_row_count": 6,
        "mini_summary": "\"Weird Al\" Yankovic has been involved in films and television shows including \"Tapeheads\", \"The Naked Gun : From the Files of Police Squad !\", \"Naked Gun 33\u2153 : The Final Insult\", \"Spy Hard\", and \"Safety Patrol\". These titles represent a selection of the media appearances in the dataset."
      },
      {
        "sub_question": "What are the different types of media that 'Weird Al' Yankovic has been involved in?",
        "attempts": [
          {
            "sql": "SELECT DISTINCT Title FROM T2 WHERE Role = 'Himself'",
            "execution_status": true,
            "error": null,
            "row_count": 3
          }
        ],
        "final_sql": "SELECT DISTINCT Title FROM T2 WHERE Role = 'Himself'",
        "final_execution_status": true,
        "final_row_count": 3,
        "mini_summary": "The different types of media that 'Weird Al' Yankovic has been involved in include television shows. Specifically, the preview shows involvement in at least three television shows: \"Space Ghost Coast to Coast\", \"The Drew Carey Show\", and \"How I Met Your Mother\"."
      },
      {
        "sub_question": "What are the notable collaborations and patterns in 'Weird Al' Yankovic's filmography and media appearances?",
        "attempts": [
          {
            "sql": "SELECT Title, Role FROM T1 WHERE Role = 'Himself'",
            "execution_status": true,
            "error": null,
            "row_count": 6
          }
        ],
        "final_sql": "SELECT Title, Role FROM T1 WHERE Role = 'Himself'",
        "final_execution_status": true,
        "final_row_count": 6,
        "mini_summary": "\"Weird Al\" Yankovic has appeared as himself in films such as \"Tapeheads,\" \"The Naked Gun : From the Files of Police Squad !,\" and \"Naked Gun 33\u2153 : The Final Insult.\" He has also appeared in at least two other films, \"Spy Hard\" and \"Safety Patrol,\" in the same capacity."
      }
    ],
    "mini_summaries": [
      "\"Weird Al\" Yankovic has been involved in films and television shows including \"Tapeheads\", \"The Naked Gun : From the Files of Police Squad !\", \"Naked Gun 33\u2153 : The Final Insult\", \"Spy Hard\", and \"Safety Patrol\". These titles represent a selection of the media appearances in the dataset.",
      "The different types of media that 'Weird Al' Yankovic has been involved in include television shows. Specifically, the preview shows involvement in at least three television shows: \"Space Ghost Coast to Coast\", \"The Drew Carey Show\", and \"How I Met Your Mother\".",
      "\"Weird Al\" Yankovic has appeared as himself in films such as \"Tapeheads,\" \"The Naked Gun : From the Files of Police Squad !,\" and \"Naked Gun 33\u2153 : The Final Insult.\" He has also appeared in at least two other films, \"Spy Hard\" and \"Safety Patrol,\" in the same capacity."
    ],
    "final_summary": "\"Weird Al\" Yankovic has had a diverse career in film and television, with notable appearances in a range of productions. His filmography includes titles such as \"Tapeheads,\" \"The Naked Gun : From the Files of Police Squad !,\" \"Naked Gun 33\u2153 : The Final Insult,\" \"Spy Hard,\" and \"Safety Patrol.\" In addition to his film work, Yankovic has also made appearances on television, including at least three shows: \"Space Ghost Coast to Coast,\" \"The Drew Carey Show,\" and \"How I Met Your Mother.\" Notably, he has often portrayed himself in these productions, appearing as himself in films such as \"Tapeheads,\" \"The Naked Gun : From the Files of Police Squad !,\" and \"Naked Gun 33\u2153 : The Final Insult,\" as well as in \"Spy Hard\" and \"Safety Patrol.\"",
    "tables": [
      "T1",
      "T2",
      "T3"
    ],
    "ground_truth": {
      "table_uids": [
        "T1",
        "T2",
        "T3"
      ]
    },
    "generated_sql": "SELECT Title FROM T1 WHERE Role = 'Himself'",
    "execution_status": true,
    "result": {
      "validation": "Success",
      "row_count": 6,
      "preview": [
        {
          "Title": "Tapeheads"
        },
        {
          "Title": "The Naked Gun : From the Files of Police Squad !"
        },
        {
          "Title": "Naked Gun 33\u2153 : The Final Insult"
        },
        {
          "Title": "Spy Hard"
        },
        {
          "Title": "Safety Patrol"
        }
      ]
    },
    "schema_mapping": {
      "T1": {
        "Year": "Year",
        "Title": "Title",
        "Role": "Role"
      },
      "T2": {
        "Year": "Year",
        "Title": "Title",
        "Role": "Role",
        "Notes": "Notes"
      },
      "T3": {
        "Year": "Year",
        "Title": "Title",
        "Role": "Role",
        "Notes": "Notes"
      }
    },
    "llm_model": "llama-3.1-8b-instant",
    "upstream_model": null
  }


---

## 6. One-Paragraph Summary (For Abstract or Intro)

*Stage 3 turns Stage 2 DPRs into grounded, judgeable answers. It decomposes each DPR into atomic sub-questions, builds an in-memory SQLite database from table metadata and sample rows (T1..T10 JSONs), and runs an agentic loop per sub-question: the LLM generates SQL using schema and example rows, the pipeline blocks cartesian joins and executes queries, and on error or empty result it refines the SQL (up to 3 attempts). Successful results are summarized into mini-summaries and then into a single final summary per DPR. This design enforces schema and data grounding and reduces join/schema/data hallucination, so Stage 4 can reliably score how well the final summary reflects the DPR given the actual data.*

---

## 7. File and Config Reference

- **Code:** `src/sql_grounding/pipeline.py`
- **Output:** `stage3_output_final.json` (or path passed via `-o`)
- **Run:**  python src/sql_grounding/pipeline.py -i "data/stage2/output/run_test_groq/dprs.json" -o "stage3_output_final.json" -n 5 --tables-meta "data/stage1/tables_clean" --require-non-empty 
  
- **Constants:** `MAX_SAMPLE_ROWS_PER_TABLE` (rows inserted per table), `LLM_SAMPLE_ROWS_PER_TABLE` (rows shown in prompt per table).
