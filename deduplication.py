import os
import json
import hashlib
import logging
import argparse
from rensa import RMinHash, RMinHashLSH
import jsonlines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

class Deduplication:
    def __init__(self, threshold=0.8):
        self.duplicate_count = 0
        self.unique_conversations = set()  
        self.threshold = threshold         
        self.lsh = None                    

    def perform_sha256_deduplication(self, input_file, output_file, update_status, update_progress):
        """Exact deduplication using SHA-256 hashing."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                reader = jsonlines.Reader(f_in)
                total_lines = sum(1 for _ in f_in)
                f_in.seek(0)
                for i, conversation in enumerate(reader):
                    update_progress(i, total_lines)
                    conversation_id = self.generate_sha256_hash(conversation)
                    if conversation_id in self.unique_conversations:
                        self.duplicate_count += 1
                        continue
                    self.unique_conversations.add(conversation_id)
                    f_out.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                update_status(f"Deduplication complete. Output file: {output_file}")
        except Exception as e:
            logging.error(f"Error during SHA-256 deduplication: {e}", exc_info=True)

    def perform_min_hash_deduplication(self, input_file, output_file, update_status, update_progress):
        try:
            num_perm = 128
            num_bands = self._calc_optimal_bands(num_perm, self.threshold)
            
            self.lsh = RMinHashLSH(threshold=self.threshold, num_perm=num_perm, num_bands=num_bands)
            
            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                reader = jsonlines.Reader(f_in)
                total_lines = sum(1 for _ in f_in)
                f_in.seek(0)
                for i, conversation in enumerate(reader):
                    update_progress(i, total_lines)
                    conversation_text = ''.join(turn['value'] for turn in conversation.get('conversations', []) if turn.get('value') is not None)
                    
                    m = self.generate_min_hash(conversation_text)
                    
                    if not self.lsh.query(m):
                        self.lsh.insert(i, m)  
                        f_out.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                    else:
                        self.duplicate_count += 1
                update_status(f"Deduplication complete. Output file: {output_file}")
        except Exception as e:
            logging.error(f"Error during MinHash deduplication: {e}", exc_info=True)

    def _calc_optimal_bands(self, num_perm, threshold):
        import math
        rows = int(num_perm / math.ceil(num_perm * threshold))
        if rows == 0:
            rows = 1
        return math.ceil(num_perm / rows)

    @staticmethod
    def generate_sha256_hash(conversation):
        conversation_text = ''.join(turn['value'] for turn in conversation.get('conversations', []) if turn.get('value') is not None)
        return hashlib.sha256(conversation_text.encode('utf-8')).hexdigest()

    @staticmethod
    def generate_min_hash(text):
        m = RMinHash(num_perm=128, seed=42)
        tokens = text.split()
        m.update(tokens)
        return m

def update_status(message):
    print(message)

def update_progress(current, total):
    progress = (current / total) * 100 if total > 0 else 100
    print(f"Progress: {progress:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Deduplication tool for conversations")
    parser.add_argument('input', help="Input JSONL file with conversations")
    parser.add_argument('output_dir', help="Directory to save deduplicated output")
    parser.add_argument('--method', choices=['sha256', 'minhash'], default='sha256', help="Deduplication method to use (default: sha256)")
    parser.add_argument('--threshold', type=float, default=0.8, help="Threshold for MinHash deduplication (default: 0.8)")
    args = parser.parse_args()

    dedup = Deduplication(threshold=args.threshold)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_file = os.path.join(args.output_dir, f"{base_name}_deduplicated.jsonl")
    
    if args.method == 'sha256':
        dedup.perform_sha256_deduplication(args.input, output_file, update_status, update_progress)
    elif args.method == 'minhash':
        dedup.perform_min_hash_deduplication(args.input, output_file, update_status, update_progress)

if __name__ == "__main__":
    main()