import os
import pandas as pd
import argparse

def combine_parquet_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.basename(input_dir)}.jsonl")
    
    with open(output_file, "w") as out_f:
        for file in os.listdir(input_dir):
            if file.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(input_dir, file))
                df.to_json(out_f, orient="records", lines=True)
    
    print(f"Converted all Parquet files in '{input_dir}' to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine Parquet files into a single JSONL file")
    parser.add_argument("input", help="Directory containing Parquet files")
    parser.add_argument("output_dir", help="Directory to save the combined JSONL file")
    
    args = parser.parse_args()
    combine_parquet_files(args.input, args.output_dir)
