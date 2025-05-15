import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
import os

MODEL_NAME = "HuggingFaceTB/fineweb-edu-classifier"
TOKENIZER_MAX_LENGTH = 512

def initialize_classifier():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, device_map="auto")
        model.eval()
        print(f"model '{MODEL_NAME}' loaded")
        return model, tokenizer
    except Exception as e:
        print(f"Error: {e}")
        exit()

def process_batch(batch, model, tokenizer, accelerator):
    for item in batch:
        if "conversations" not in item or not isinstance(item["conversations"], list):
            item["score"] = 0.0
        else:
            inputs = tokenizer(
                [mes["value"] for mes in item["conversations"]],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=TOKENIZER_MAX_LENGTH
            )
            inputs = {k: v.to(device=accelerator.device) for k,v in inputs.items()}
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            except Exception as e:
                print(inputs)
                print(inputs["input_ids"].shape)
                print(inputs["token_type_ids"].shape)
                print(inputs["attention_mask"].shape)
                raise e
            item["score"] = np.mean(outputs.logits.squeeze(-1).float().cpu().numpy()).item()
    return batch

# So i dont' have to pass through dataset converter again
def clean_output(obj):
    if not obj or "conversations" not in obj:
        return obj
    clean_obj = {"conversations": []}
    for turn in obj["conversations"]:
        if turn.get("from") in ["human", "gpt", "system"]:
            clean_turn = {
                "from": turn["from"],
                "value": turn["value"]
            }
            clean_obj["conversations"].append(clean_turn)
    return clean_obj

def main():
    accelerator = Accelerator()
    parser = argparse.ArgumentParser(
        description="use finewebedu's classifer to clean instruct data"
    )
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.is_file():
        accelerator.print(f"Retard: file not found at '{input_path}'")
        return
    
    output_dir_path = Path(args.output_dir)
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        accelerator.print(f"Error : {e}")
        return
    
    model, tokenizer = initialize_classifier()
    dataset = load_dataset("json", data_files=str(input_path), split="train")
    
    # increase bsz later
    def collate_fn(data):
        return data
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    dataloader, model, tokenizer = accelerator.prepare(dataloader, model, tokenizer)
    
    base_name = input_path.stem
    process_idx = f"_{accelerator.process_index}" if accelerator.num_processes > 1 else ""
    rated_file_path = output_dir_path / f"{base_name}_rated{process_idx}.jsonl"
    
    total_processed = 0
    successful_ratings = 0
    
    with open(rated_file_path, 'w', encoding='utf-8') as f_out:
        for batch in tqdm(dataloader, desc="Rating conversations"):
            processed_obj = process_batch(batch, model, tokenizer, accelerator)
            if processed_obj:
                for obj in processed_obj:
                    f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                successful_ratings += len(batch)
            total_processed += len(batch)
    
    accelerator.wait_for_everyone()
    
    # Gather all ratings on the main process
    if accelerator.is_main_process:
        all_ratings = []
        for file in os.listdir(output_dir_path):
            if file.startswith(f"{base_name}_rated"):
                with open(output_dir_path / file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            if "score" in obj:
                                all_ratings.append(obj["score"])
                        except:
                            continue
        
        print(f"\nRating stats ({len(all_ratings)} processed):")
        print(f"Min: {min(all_ratings):.2f}, Max: {max(all_ratings):.2f}, Mean: {np.mean(all_ratings):.2f}")
        
        thresholds = [i * 0.5 for i in range(11)]
        print("\nRape report (threshold -> items kept):")
        print("-----------------------------------------")
        for threshold in thresholds:
            kept = sum(1 for r in all_ratings if r >= threshold)
            percentage = 100 * kept / len(all_ratings)
            print(f"≥ {threshold:.1f}: {kept} items ({percentage:.1f}%)")
        
        # Added filtering prompt here
        threshold_input = input("\nFilter data? (y/n): ").strip().lower()
        if threshold_input == 'y':
            while True:
                try:
                    threshold = float(input("Enter threshold: ").strip())
                    if 0 <= threshold <= 5:
                        break
                    print("Enter a value between 0 and 5")
                except ValueError:
                    print("Invalid input")
        else:
            threshold = None
    else:
        threshold = None

    # Use Python's file system to communicate the threshold
    if accelerator.is_main_process and threshold is not None:
        with open(output_dir_path / "threshold.txt", "w") as f:
            f.write(str(threshold))
    
    accelerator.wait_for_everyone()
    
    # All processes read the threshold from the file
    if not accelerator.is_main_process and os.path.exists(output_dir_path / "threshold.txt"):
        with open(output_dir_path / "threshold.txt", "r") as f:
            try:
                threshold = float(f.read().strip())
            except:
                threshold = None
    
    # Filter based on threshold
    if threshold is not None:
        filtered_file = output_dir_path / f"{base_name}_filtered_{threshold:.1f}{process_idx}.jsonl"
        with open(rated_file_path, 'r', encoding='utf-8') as f_in, \
             open(filtered_file, 'w', encoding='utf-8') as f_out:
            kept_count = 0
            for line in tqdm(f_in, desc=f"Filtering (threshold ≥ {threshold:.1f})"):
                try:
                    obj = json.loads(line)
                    if obj.get("score", -1) >= threshold:
                        clean_obj = clean_output(obj)
                        f_out.write(json.dumps(clean_obj, ensure_ascii=False) + '\n')
                        kept_count += 1
                except:
                    continue
            print(f"Saved {kept_count} filtered items")

    # Clean up the threshold file
    if accelerator.is_main_process and os.path.exists(output_dir_path / "threshold.txt"):
        os.remove(output_dir_path / "threshold.txt")

if __name__ == "__main__":
    main()