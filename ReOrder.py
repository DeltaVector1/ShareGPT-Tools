import json
import argparse
import os

def reorder_fields(json_obj):
    """Reorder fields in a JSON object with special handling for system messages."""
    # Handle conversation array
    if "conversations" in json_obj:
        ordered_conversations = []
        for conv in json_obj["conversations"]:
            if "from" in conv and "value" in conv:
                ordered_conv = {
                    "from": conv["from"],
                    "value": conv["value"],
                    "loss": conv.get("loss", True)
                }
                
                # Only add prefix if not a system message
                if conv["from"] != "system" and "prefix" in conv:
                    ordered_conv["prefix"] = conv["prefix"]
                
                ordered_conversations.append(ordered_conv)
        return {"conversations": ordered_conversations}
    
    # Handle single conversation object
    elif "from" in json_obj and "value" in json_obj:
        ordered_conv = {
            "from": json_obj["from"],
            "value": json_obj["value"],
            "loss": json_obj.get("loss", True)
        }
        
        # Only add prefix if not a system message
        if json_obj["from"] != "system" and "prefix" in json_obj:
            ordered_conv["prefix"] = json_obj["prefix"]
            
        return ordered_conv
    else:
        print(f"Warning: Malformed object without required fields: {json_obj}")
        return json_obj

def process_jsonl(input_file, output_dir):
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                if line.strip():  # Skip empty lines
                    json_obj = json.loads(line.strip())
                    reordered_obj = reorder_fields(json_obj)
                    outfile.write(json.dumps(reordered_obj) + "\n")
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: Invalid JSON - {e}")
            except Exception as e:
                print(f"Line {line_num}: Error processing - {e}")

    print(f"Processed file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and reorder ShareGPT JSONL files.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output file")

    args = parser.parse_args()
    process_jsonl(args.input, args.output_dir)