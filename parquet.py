import pandas as pd
import json
import argparse
import os
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def parquet_to_jsonl(input_path: str, output_path: str):
    print(f"Reading parquet file: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Add directory creation
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting to JSONL: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json_line = row.to_dict()
            f.write(json.dumps(json_line, ensure_ascii=False, cls=NumpyEncoder) + '\n')

    print(f"Converted {len(df)} rows to JSONL")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Parquet to JSONL")
    parser.add_argument("input", help="Input parquet file")
    parser.add_argument("output_dir", help="Directory to save the output JSONL file")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output file path
    output_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.input))[0]}.jsonl")
    
    parquet_to_jsonl(args.input, output_file)
