import json
import argparse
import os
import glob
from typing import List, Dict, Any
from transformers import AutoTokenizer
from tqdm import tqdm

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {file_path}")
    return data
    
def count_tokens(data, tokenizer):
    """Count tokens only in GPT responses using the fast tokenizer with tqdm logging per row."""
    total_tokens = 0
    for item in tqdm(data, desc="Tokenizing rows"):
        if not isinstance(item, dict) or "conversations" not in item:
            continue
        for turn in item["conversations"]:
            if turn.get("from") != "gpt" or "value" not in turn:
                continue
            text = turn["value"]
            if not isinstance(text, str):
                continue
            try:
                token_count = len(tokenizer(text)['input_ids'])  # Uses fast tokenizer
                total_tokens += token_count
            except Exception as e:
                print(f"Warning: Error tokenizing text: {str(e)[:100]}")
    return total_tokens

def process_folder(input_dir: str, tokenizer, pattern: str = "*.jsonl") -> int:
    """Process all JSONL files in a folder and return GPT token count."""
    input_pattern = os.path.join(input_dir, pattern)
    input_files = sorted(glob.glob(input_pattern))
    if not input_files:
        print(f"No files matching pattern {pattern} found in {input_dir}")
        return 0
    print(f"Found {len(input_files)} files to process")

    total_tokens = 0
    for input_file in tqdm(input_files, desc="Processing files"):
        try:
            data = load_jsonl(input_file)
            total_tokens += count_tokens(data, tokenizer)
        except Exception as e:
            print(f"Error processing file {input_file}: {str(e)[:100]}")
    return total_tokens

def main():
    parser = argparse.ArgumentParser(description="Count GPT tokens in ShareGPT format")
    parser.add_argument("input", help="Directory containing input JSONL files")
    parser.add_argument("output_dir", nargs="?", help="Directory to save output summary (optional)")
    parser.add_argument("--tokenizer", default="gpt2", help="HuggingFace tokenizer to use (default: gpt2)")
    parser.add_argument("--pattern", default="*.jsonl", help="File pattern to match (default: *.jsonl)")
    args = parser.parse_args()

    print(f"Loading {args.tokenizer} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    total_tokens = process_folder(args.input, tokenizer, args.pattern)

    print("\nToken Count Summary:")
    print(f"Total GPT tokens: {total_tokens:,}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "token_count_summary.json")
        summary = {"total_gpt_tokens": total_tokens}
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {output_file}")

if __name__ == "__main__":
    main()
