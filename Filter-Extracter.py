import json
import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Filter jsonl file based on score threshold")
    parser.add_argument('input_file', type=str, help="Input rated jsonl file")
    parser.add_argument('threshold', type=float, help="Threshold value for filtering")
    parser.add_argument('output_file', type=str, help="Output filtered file")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    threshold = args.threshold
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return
    
    total = 0
    kept = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Filtering"):
            total += 1
            try:
                obj = json.loads(line)
                if obj.get("score", -1) >= threshold:
                    # Remove the score field
                    if "score" in obj:
                        del obj["score"]
                    
                    # Clean conversations format if needed
                    if "conversations" in obj:
                        clean_conversations = []
                        for turn in obj["conversations"]:
                            if turn.get("from") in ["human", "gpt", "system"]:
                                clean_turn = {
                                    "from": turn["from"],
                                    "value": turn["value"]
                                }
                                clean_conversations.append(clean_turn)
                        obj["conversations"] = clean_conversations
                    
                    f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    kept += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")
                continue
    
    print(f"Done! Kept {kept}/{total} lines ({kept/total*100:.1f}%)")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()