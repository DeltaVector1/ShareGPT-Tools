import json
import argparse
from pathlib import Path

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line}. Error: {e}")
    return data

def load_filter_criteria(filter_files):
    filter_criteria = []
    for filter_file in filter_files:
        with open(filter_file, 'r', encoding='utf-8', errors='replace') as f:
            filter_criteria.extend(line.strip() for line in f if line.strip())
    return filter_criteria

def filter_conversations(conversations, filter_criteria, threshold=None):
    filtered_conversations = []
    total_matched_phrases = 0
    removed_conversation_count = 0
    matched_counts = []

    for conversation in conversations:
        total_phrases_in_conversation = 0
        for msg in conversation.get("conversations", []):
            if msg.get("from") == "gpt" and msg.get("value"):
                matched_phrases = [phrase for phrase in filter_criteria if phrase in msg["value"]]
                total_phrases_in_conversation += len(matched_phrases)
        matched_counts.append(total_phrases_in_conversation)
        total_matched_phrases += total_phrases_in_conversation

    if total_matched_phrases == 0:
        return conversations, 0, total_matched_phrases, 0

    average_matched_phrases = total_matched_phrases / len(conversations) if conversations else 0
    for idx, conversation in enumerate(conversations):
        if threshold is not None and matched_counts[idx] >= average_matched_phrases * threshold:
            removed_conversation_count += 1
        else:
            filtered_conversations.append(conversation)

    return filtered_conversations, removed_conversation_count, total_matched_phrases, removed_conversation_count

def write_filtered_jsonl(filtered_data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8', errors='replace') as file:
        for conversation in filtered_data:
            json.dump(conversation, file, ensure_ascii=False)
            file.write('\n')

def filter_dataset(dataset_file, filter_files, threshold=None):
    filter_criteria = load_filter_criteria(filter_files)
    data = load_jsonl(dataset_file)
    filtered_data, filtered_count, total_matched_phrases, removed_conversation_count = filter_conversations(data, filter_criteria, threshold)

    output_folder = Path(__file__).parent / "deslopped"
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_name = Path(dataset_file).stem
    output_file_path = output_folder / f"{dataset_name}_deslopped.jsonl"
    write_filtered_jsonl(filtered_data, output_file_path)

    output_message = (
        f"Total original conversations: {len(data)}\n"
        f"Total filtered conversations: {filtered_count}\n"
        f"Remaining conversations after filtering: {len(filtered_data)}\n"
        f"Total matched phrases: {total_matched_phrases}\n"
        f"Total conversations removed: {removed_conversation_count}\n"
        f"Filtered output written to: {output_file_path}\n"
    )
    return output_message

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter conversations based on specific criteria.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset JSONL file.")
    parser.add_argument("--filters", required=True, nargs="+", help="Path to one or more filter files.")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold multiplier for filtering.")
    
    args = parser.parse_args()
    print(filter_dataset(args.dataset, args.filters, args.threshold))