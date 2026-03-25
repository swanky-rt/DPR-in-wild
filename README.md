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

## 2. End-to-End Flow 

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

