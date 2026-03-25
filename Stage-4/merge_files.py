import json

files = [
    "stage3_output_batch1.json",
    "stage3_output_batch2.json",
    "stage3_output_batch3.json",
    "stage3_output_batch4.json",
    "stage3_output_batch5.json",
    "stage3_output_batch6.json",
]

merged = []
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} is not a JSON list")
        merged.extend(data)

with open("stage3_output_final.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(merged)} DPRs into stage3_output_final.json")