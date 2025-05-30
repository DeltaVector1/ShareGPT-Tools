import os
import json
import hashlib
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from functools import partial
from rensa import RMinHash
import jsonlines
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

class Deduplication:
    def __init__(self, threshold=0.8, num_processes=None):
        self.duplicate_count = 0
        self.unique_hashes = set()
        self.threshold = threshold
        self.num_processes = num_processes or max(1, mp.cpu_count() - 1)
        self.num_perm = 128

    def process_chunk(self, chunk, method):
        results = []
        local_unique_hashes = set()

        for conversation in chunk:
            conversation_text = ''.join(
                turn['value']
                for turn in conversation.get('conversations', [])
                if turn.get('value') is not None
            )

            if method == 'sha256':
                hash_value = self.generate_sha256_hash(conversation_text)
                if hash_value not in self.unique_hashes and hash_value not in local_unique_hashes:
                    local_unique_hashes.add(hash_value)
                    results.append(conversation)

            elif method == 'minhash':
                minhash = self.generate_rminhash(conversation_text)
                hash_tuple = tuple(minhash.digest())
                if hash_tuple not in self.unique_hashes and hash_tuple not in local_unique_hashes:
                    local_unique_hashes.add(hash_tuple)
                    results.append(conversation)

        return results, local_unique_hashes

    def perform_deduplication(self, input_file, output_dir, method='sha256'):
        try:
            output_file = os.path.join(output_dir, f"deduplicated_{os.path.basename(input_file)}")
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            with jsonlines.open(input_file) as reader:
                data = list(reader)

            chunk_size = max(100, len(data) // (self.num_processes * 10))
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

            with mp.Pool(self.num_processes) as pool:
                process_func = partial(self.process_chunk, method=method)
                processed_results = list(tqdm(
                    pool.imap(process_func, chunks),
                    total=len(chunks),
                    desc=f"Processing with {method}"
                ))

            unique_data = []
            for results, local_hashes in processed_results:
                for hash_val in local_hashes:
                    if hash_val not in self.unique_hashes:
                        self.unique_hashes.add(hash_val)
                    else:
                        self.duplicate_count += 1

            for results, _ in processed_results:
                for conversation in results:
                    conversation_text = ''.join(
                        turn['value']
                        for turn in conversation.get('conversations', [])
                        if turn.get('value') is not None
                    )
                    if method == 'sha256':
                        hash_val = self.generate_sha256_hash(conversation_text)
                    else:
                        minhash = self.generate_rminhash(conversation_text)
                        hash_val = tuple(minhash.digest())

                    if hash_val in self.unique_hashes:
                        unique_data.append(conversation)
                        self.unique_hashes.remove(hash_val)

            self.duplicate_count = len(data) - len(unique_data)

            with jsonlines.open(output_file, mode='w') as writer:
                writer.write_all(unique_data)

            logging.info(f"Deduplication complete. Removed {self.duplicate_count} duplicates. Output: {output_file}")
            return output_file

        except Exception as e:
            logging.error(f"Error during {method} deduplication: {e}", exc_info=True)
            raise

    @staticmethod
    def generate_sha256_hash(text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def generate_rminhash(self, text):
        minhash = RMinHash(num_perm=self.num_perm, seed=42)
        minhash.update(text.split())
        return minhash

def main():
    parser = argparse.ArgumentParser(description="Deduplication tool for conversations")
    parser.add_argument('input_file', type=str, help="Input JSONL file with conversations")
    parser.add_argument('output_dir', type=str, help="Output directory for deduplicated conversations")
    parser.add_argument('--method', choices=['sha256', 'minhash'], default='sha256',
                        help="Deduplication method to use (default: sha256)")
    parser.add_argument('--threshold', type=float, default=0.8,
                        help="Threshold for MinHash similarity (default: 0.8)")
    parser.add_argument('--processes', type=int, default=None,
                        help="Number of processes to use (default: CPU count - 1)")

    args = parser.parse_args()

    dedup = Deduplication(threshold=args.threshold, num_processes=args.processes)
    dedup.perform_deduplication(args.input_file, args.output_dir, method=args.method)

if __name__ == "__main__":
    main()
