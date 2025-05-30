#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import functools

def is_line_not_empty(line):
  return functools.reduce(lambda acc, char: acc or char != ' ', False, line) and len(line) > 0
    
def clean(data):
    for item in data:
        conversations = item.get("conversations", [])
        filtered = [
            msg for msg in conversations 
            if not (msg.get("from") == "gpt" and msg.get("value") == "")
        ]
        item["conversations"] = filtered
    return data



def is_line_empty(line):
  try:
    return all(ord(c) == 0x20 for c in line) or len(line) == 0
  except TypeError:
    return True

def main():
    parser = argparse.ArgumentParser(description="Remove empty GPT turns from ShareGPT dataset")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_file", help="Path to output JSONL file")
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read input file
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Clean data
    cleaned_data = clean_empty_gpt_turns(data)
    
    # Write cleaned data to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Processed {len(data)} entries. Output written to {output_path}")

if __name__ == "__main__":
    main()