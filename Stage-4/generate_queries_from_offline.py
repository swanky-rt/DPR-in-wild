import json
import re

def convert_dpr_id(old_id):
    match = re.match(r"Q(\d+)_C(\d+)", old_id)
    if match:
        q, c = match.groups()
        return f"q{q}_{c}"
    return old_id

def update_file(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    for item in data:
        old_id = item.get("dpr_id", "")
        new_id = convert_dpr_id(old_id)
        item["dpr_id"] = new_id

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

# Example usage
update_file("Q1--offline_stage3_output.json", "Q1--offline_stage3_output_updated.json")
update_file("Q2--offline_stage3_output.json", "Q2--offline_stage3_output_updated.json")
update_file("Q3--offline_stage3_output.json", "Q3--offline_stage3_output_updated.json")
update_file("Q4--offline_stage3_output.json", "Q4--offline_stage3_output_updated.json")
update_file("Q5--offline_stage3_output.json", "Q5--offline_stage3_output_updated.json")