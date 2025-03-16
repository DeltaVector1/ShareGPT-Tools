import json
import os
import argparse
from fuzzywuzzy import fuzz

class DatasetConverter:
    @staticmethod
    def load_data(input_path: str) -> list:
        ext = os.path.splitext(input_path)[1].lower()
        if ext == '.json':
            return DatasetConverter.load_json_data(input_path)
        elif ext == '.jsonl':
            return DatasetConverter.load_jsonl_data(input_path)
        else:
            raise ValueError("Unsupported file format")

    @staticmethod
    def load_json_data(input_path: str) -> list:
        data = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                try:
                    data = json.loads(file_content)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    print("JSON Decode Error. Attempting to process line by line.")
                    lines = file_content.splitlines()
                    for line in lines:
                        line = line.strip()
                        if line:
                            try:
                                json_object = json.loads(line)
                                if isinstance(json_object, dict):
                                    data.append(json_object)
                            except json.JSONDecodeError:
                                print(f"Skipping invalid JSON line: {line}")
                                data.extend(DatasetConverter.fallback_parse_line(line))
        except UnicodeDecodeError:
            print("Unicode Decode Error. Ensure file is encoded in UTF-8.")
        return data

    @staticmethod
    def load_jsonl_data(input_path: str) -> list:
        data = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line: {line}")
                            data.extend(DatasetConverter.fallback_parse_line(line))
        except UnicodeDecodeError:
            print("Unicode Decode Error. Ensure file is encoded in UTF-8.")
        return data

    @staticmethod
    def extract_conversations(entry: dict) -> list:
        conversations = []
        if 'conversations' in entry:
            for message in entry['conversations']:
                role = message.get('from')
                if role == 'user':
                    role = 'human'
                conv_entry = {
                    "from": role if role != 'assistant' else 'gpt',
                    "value": message.get('value', '') if message.get('value') else ''                }
                if 'prefix' in message:
                    conv_entry['prefix'] = message['prefix']
                if 'loss' in message:
                    conv_entry['loss'] = message['loss']
                conversations.append(conv_entry)
        else:
            if 'system' in entry:
                conversations.append({"from": "system", "value": entry['system'].strip()})
            if 'completion' in entry:
                DatasetConverter.process_completion(entry['completion'], conversations)
            elif 'messages' in entry:
                for message in entry.get('messages', []):
                    if isinstance(message, dict):
                        role = message.get('role')
                        if role == 'user':
                            role = 'human'
                        elif role == 'assistant':
                            role = 'gpt'
                        conv_entry = {
                            "from": role,
                            "value": message.get('content', '').strip()
                        }
                        if 'prefix' in message:
                            conv_entry['prefix'] = message['prefix']
                        if 'loss' in message:
                            conv_entry['loss'] = message['loss']
                        conversations.append(conv_entry)
        if not conversations:
            return [{"from": "system", "value": "No conversations found."}]
        return conversations

    @staticmethod
    def process_completion(completion: dict, conversations: list):
        if isinstance(completion, list):
            for message in completion:
                DatasetConverter.add_conversation(message, conversations)
        elif isinstance(completion, str):
            try:
                completion_json = json.loads(completion)
                if isinstance(completion_json, list):
                    for message in completion_json:
                        DatasetConverter.add_conversation(message, conversations)
            except json.JSONDecodeError:
                pass

    @staticmethod
    def add_conversation(message: dict, conversations: list):
        role = message.get('role')
        if role == 'user':
            role = 'human'
        elif role == 'assistant':
            role = 'gpt'
        conv_entry = {
            "from": role,
            "value": message.get('content', '').strip()
        }
        if 'prefix' in message:
            conv_entry['prefix'] = message['prefix']
        if 'loss' in message:
            conv_entry['loss'] = message['loss']
        conversations.append(conv_entry)

    @staticmethod
    def fallback_parse_line(line: str) -> list:
        conversations = []
        keywords = {'system': 'system:', 'user': 'user:', 'assistant': 'assistant:'}
        for role, keyword in keywords.items():
            if keyword in line:
                value = line.split(keyword, 1)[1].strip()
                conversations.append({"from": role if role != 'assistant' else 'gpt', "value": value})
        if not conversations:
            potential_roles = ['system', 'user', 'assistant']
            for role in potential_roles:
                ratio = fuzz.ratio(line.lower(), role)
                if ratio > 70:
                    conversations.append({"from": role if role != 'assistant' else 'gpt', "value": line.strip()})
                    break
        if not conversations:
            conversations.append({"from": "unknown", "value": line.strip()})
        return conversations

    @staticmethod
    def validate_jsonl(output_path: str):
        with open(output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON at line {i}: {line}")
                        raise ValueError(f"Invalid JSONL format detected at line {i}.")
        print("Validation completed: The output is proper JSONL.")

    @staticmethod
    def process_data(data: list, output_path: str) -> list:
        preview_entries = []
        conversations_found = False
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                conversations = DatasetConverter.extract_conversations(entry)
                formatted_entry = {"conversations": conversations}
                f.write(json.dumps(formatted_entry, ensure_ascii=False) + '\n')
                conversations_found = True
                if len(preview_entries) < 3:
                    preview_entries.append(formatted_entry)
        status_message = "Conversations completed successfully." if conversations_found else "No conversations found for this dataset."
        print(status_message)
        DatasetConverter.validate_jsonl(output_path)
        return preview_entries

    @staticmethod
    def process_multiple_files(input_paths: list, output_dir: str) -> dict:
        preview_entries = {}
        for input_path in input_paths:
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jsonl")
            print(f"Processing file: {filename}")
            data = DatasetConverter.load_data(input_path)
            preview = DatasetConverter.process_data(data, output_path)
            preview_entries[filename] = preview
        return preview_entries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON/JSONL datasets into structured conversation format.")
    parser.add_argument("input", nargs="+", help="Input file paths (JSON/JSONL)")
    parser.add_argument("output_dir", help="Output directory for the processed files")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    converter = DatasetConverter()
    preview = converter.process_multiple_files(args.input, args.output_dir)
    print("Preview of processed conversations:", json.dumps(preview, indent=2, ensure_ascii=False))
