import os
import pandas as pd

input_dir = "/home/mango/Misc/Scripts/ShareGPT-Tools/r1"
output_file = "R1.jsonl"

with open(output_file, "w") as out_f:
    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(input_dir, file))
            df.to_json(out_f, orient="records", lines=True)

print(f"Converted all Parquet files in '{input_dir}' to '{output_file}'")
