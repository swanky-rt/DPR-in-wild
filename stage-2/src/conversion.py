import json

with open("src/query_table_cluster_matches.json") as f:
    data = json.load(f)

output = []
for q in data["query_results"][:5]:
    output.append({
        "dpr_id": q["query_id"],
        "user_query": q["query_text"],
        "matched_local_table_ids": q["matched_local_table_ids_from_input"]
    })

with open("data/user_queries_corrected.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Done: {len(output)} queries written")