import json
from pathlib import Path


current_dir = Path.cwd()
output_dir = current_dir.parent.parent.parent / "Stage-4"
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(
    f for f in current_dir.glob("stage3_output_batch*.json")
    if "execution_summary" not in f.name
)

output_file = output_dir / "stage3_output_final.json"
merged = []
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} is not a JSON list")
        merged.extend(data)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(merged)} DPRs into {output_file}")