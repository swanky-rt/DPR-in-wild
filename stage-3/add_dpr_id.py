import json
import glob
import re
import os

# Path to the online_with_query directory
stage3_dir = "data/stage3_outputs/online_with_query"

# Find all stage3 output files (not execution_summary)
files = glob.glob(os.path.join(stage3_dir, "Q*--online_stage3_output.json"))

for filepath in sorted(files):
    filename = os.path.basename(filepath)

    # Extract query number from filename e.g. "Q3--online_stage3_output.json" -> 3
    match = re.match(r'Q(\d+)--online_stage3_output\.json', filename)
    if not match:
        print(f"Skipping {filename} — couldn't parse query number")
        continue

    query_num = int(match.group(1))

    with open(filepath, 'r') as f:
        records = json.load(f)

    # Assign dpr_id in format q1_1, q1_2, etc.
    for i, rec in enumerate(records):
        rec['dpr_id'] = f"q{query_num}_{i + 1}"

    # Overwrite the same file
    with open(filepath, 'w') as f:
        json.dump(records, f, indent=2)

    print(f"Q{query_num}: assigned {len(records)} IDs → q{query_num}_1 to q{query_num}_{len(records)}")

print("\nDone. All stage3 output files updated with dpr_ids.")