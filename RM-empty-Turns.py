#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def clean_empty_gpt_turns(data):
    """Remove empty GPT turns from conversations."""
    for item in data:
        conversations = item.get("conversations", [])
        # Filter out any GPT messages with empty value
        filtered_conversations = [
            msg for msg in conversations 
            if not (msg.get("from") == "gpt" and msg.get("value") == "")
        ]
        item["conversations"] = filtered_conversations
    return data

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