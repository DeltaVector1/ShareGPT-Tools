import json

def convert_to_sharegpt(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            
            # Create ShareGPT format
            sharegpt_format = {
                "conversations": [
                    {"from": "human", "value": data["messages"][0]["content"]},
                    {"from": "gpt", "value": ""}  # Empty response since ground_truth is metadata
                ]
            }
            
            # If there are additional messages in the original format, add them
            for msg in data["messages"][1:]:
                role = "human" if msg["role"] == "user" else "gpt"
                sharegpt_format["conversations"].append({"from": role, "value": msg["content"]})
            
            f_out.write(json.dumps(sharegpt_format) + '\n')

# Usage
convert_to_sharegpt('nemotron_extra_sharegpt.jsonl', 'nemo.jsonl')