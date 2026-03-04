## Stage 1 – Data Selection, Table Preparation, Descriptions, and Embeddings (Uninformed User Baseline)

Stage 1 creates the table inputs needed for the rest of the pipeline. It selects 10 unique tables from HybridQA, keeps the column names exactly the same, generates a short description for each table, and creates one embedding vector per table for clustering. The same tables (T1–T10) are used again in later stages such as clustering, DPR generation, and SQL grounding.

---

### Input

HybridQA dataset (train split) loaded from Hugging Face: `wenhu/hybrid_qa`.

---

### How the 10 tables are chosen

1. Load the HybridQA train split.
2. Find all unique HybridQA table IDs (many QA rows share the same table).
3. For each table, look at the table title and guess a simple topic label (domain) using keywords.

   * Examples: Sports, Music, Film/TV, Geography, Politics, Science
   * If no keyword matches, label it as General.
   * If the title is missing, label it as Unknown.
4. Select tables while keeping some variety by using a small constraint:

   * Choose at most 3 tables from the same domain.
5. Stop when 10 unique tables are selected.
6. Rename them to `T1` to `T10` so downstream stages can use consistent IDs.

---

### Output files

Stage 1 produces the following files and folders.

#### 1) Raw selected tables

Folder: `tables_raw/`

* `T1.json ... T10.json`

Each file includes:

* normalized `table_id` (T1..T10)
* original HybridQA `source_table_id`
* guessed `domain`
* the raw table content (title, header, data)

Also produced:

* `table_manifest.json` (one summary entry per table, including rows/cols and paths)

#### 2) Cleaned tables with descriptions

Folder: `tables_clean/`

* `T1.json ... T10.json`

Each cleaned file includes:

* `table_id`, `source_table_id`, `domain`, `title`
* `columns` (kept exactly the same as the original header)
* `rows` (table data)
* `description` (2–4 sentences)
* `numeric_columns`, `categorical_columns`, `entities` (LLM-predicted)

Also produced:

* `schema_descriptions.json` (combined summary for all 10 tables)

#### 3) Table embeddings

File: `table_embeddings.json`

This file contains one entry per table with:

* `table_id`
* `columns`
* `description`
* `embedding` (vector; dimension 384)

---

### What happens in Stage 1 (high level)

1. Save raw tables

* Select 10 tables and write them to `tables_raw/`
* Write `table_manifest.json`

2. Generate descriptions

* Read each table from `tables_raw/`
* Keep the column names unchanged
* Use an LLM to generate the description and column groups
* Write `tables_clean/` and `schema_descriptions.json`

3. Generate embeddings

* For each table, build one text string:
  `Title: <title>. Columns: <col1, col2, ...>. Description: <description>`
* Embed the text using SentenceTransformer (`all-MiniLM-L6-v2`)
* Write `table_embeddings.json`

---

### How to run

From the project root:

1. Select and save the 10 tables:

```bash
python layer0_select_tables.py
```

2. Generate descriptions (LLM):

```bash
python layer1_descriptions.py
```

Requirements:

* `litellm` installed
* `LITELLM_API_KEY` set in the environment
* API routed via Keymaker (`https://thekeymaker.umass.edu/`)

3. Generate embeddings:

```bash
python layer1_table_embeddings.py
```

Requirements:

* `sentence-transformers` and `torch` installed
* The first run downloads `all-MiniLM-L6-v2` (one time)

---

### Notes

* The embedding vector length is 384 because `all-MiniLM-L6-v2` always returns 384-dimensional vectors.
* Stage 2 (Clustering + DPR Generation) should use:

    table_embeddings.json for semantic similarity between tables (embedding vectors)

    columns for common column overlap

    Optionally, domain, numeric_columns, categorical_columns, and entities from tables_clean/ for extra heuristics

* Later stages reuse the same T1..T10 tables:

    Stage 3 (SQL generation and grounding) uses tables_clean/ (rows + columns) to build a SQLite database and validate the generated DPRs by execution.
