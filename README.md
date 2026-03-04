# Stage 3 – SQL Generation & Grounding

This stage is responsible for validating whether the Data Product Requests (DPRs) generated in Stage 2 can actually be grounded in the original structured tables.

Input: dprs.json, Metadata: T1.json to T10.json

For each DPR:
- Load referenced tables
- Generate SQL using LLM
- Execute SQL on SQLite & Validate execution

Output: stage3_output.json

### Steps:
1. Schema Preparation
- Load relevant tables from data/stage1/tables_clean
- Build SQLite database in-memory
- Create schema mapping 

2. SQL Generation
- Provide DPR text, Table schema
- LLM generates a SQLite-compatible query 
- Using the Llama-3.1-8b model (via Groq)

3. SQL Grounding & Validation
- The generated SQL is executed on the SQLite database

### How to Run:
In the terminal, from the project root:
python src/sql_grounding/pipeline.py `
  -i "data/stage2/output/run_test_groq/dprs.json" `
  -o "stage3_output.json" `
  --tables-meta "data/stage1/tables_clean"

### Observations:
During execution-based validation, three patterns were observed:
- Executable queries with valid results – strong grounding.
- Executable queries returning zero rows – schema valid but semantically weak joins and strong filters.
- Failed execution due to missing columns – LLM schema hallucination.

### Conclusion of stage-3:
Stage 3 demonstrates that DPR generation must be coupled with execution-based grounding. Without this step, hallucinated or infeasible data products could propagate silently into systems.