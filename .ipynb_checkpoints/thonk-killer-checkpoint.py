import json
import re
import sys
from tqdm import tqdm

def remove_think_content(text):
    """Removes anything between <think> tags, including the tags themselves."""
    return re.sub(r'\n?', '', text, flags=re.DOTALL)

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Processing JSONL", unit="line"):
            try:
                data = json.loads(line)
                if "conversations" in data:
                    for item in data["conversations"]:
                        if "value" in item:
                            item["value"] = remove_think_content(item["value"])
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}", file=sys.stderr)
# Example usage
if __name__ == "__main__":
    input_filename = "unique_nemo.jsonl"
    output_filename = "final-nemo.jsonl"
    process_jsonl(input_filename, output_filename)