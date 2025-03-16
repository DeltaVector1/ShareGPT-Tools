import collections
import re
import json
import argparse
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
import string

# Attempt to download stopwords if not already available
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

# Expand stopword list (you can add more stopwords if necessary)
EXTRA_STOPWORDS = {'said', 'ask', 'tell', 'response', 'question'}  # You can add more words here
STOP_WORDS.update(EXTRA_STOPWORDS)

def tokenize(string, no_punctuation=False):
    if no_punctuation:
        string = re.sub(r'[^\w\s]', '', string)  # Removes anything that's not a word or space

    if not no_punctuation:
        words = re.findall(r'\w+|[^\w\s]', string.lower())
    else:
        words = re.findall(r'\b\w+\b', string.lower())

    return words

def count_ngrams(lines, min_length=3, max_length=5, stopword_limit=1, punctuation_limit=1, no_punctuation=False):
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}

    for line in lines:
        words = tokenize(line, no_punctuation=no_punctuation)

        if len(words) < min_length:
            continue

        for n in lengths:
            for i in range(len(words) - n + 1):
                current_slice = tuple(words[i:i + n])

                stopwords_in_ngram = sum(1 for word in current_slice if word in STOP_WORDS)
                punctuation_in_ngram = sum(1 for word in current_slice if word in string.punctuation)

                if stopwords_in_ngram > stopword_limit or punctuation_in_ngram > punctuation_limit:
                    continue

                ngrams[n][current_slice] += 1

    return ngrams

def process_jsonl(filename, role_filter, no_punctuation=False):
    try:
        with open(filename, 'r', encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)  
            f.seek(0)  

            with tqdm(total=total_lines, desc="Processing JSONL", unit="line") as pbar:
                for line in f:
                    line = line.strip()

                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]  
                        line = line.replace('\\"', '"')  

                    try:
                        json_obj = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON line: {line[:100]}...")
                        pbar.update(1)
                        continue

                    if not isinstance(json_obj, dict):
                        print(f"Skipping unexpected JSON structure: {line[:100]}...")
                        pbar.update(1)
                        continue

                    for conversation in json_obj.get("conversations", []):
                        if "value" in conversation:
                            sender = conversation.get("from", "").lower()

                            if role_filter == ['all'] or sender in role_filter:
                                yield conversation["value"]

                    pbar.update(1)

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except IOError:
        print(f"Error: Unable to read the file '{filename}'.")

def main():
    parser = argparse.ArgumentParser(description="Process a JSONL file and extract n-grams from GPT conversations.")
    parser.add_argument('input', help="Path to the JSONL file")
    parser.add_argument('output_dir', nargs='?', help="Directory to save output (optional)")
    parser.add_argument('--role_filter', type=str, nargs='+', default=['gpt'], choices=['gpt', 'system', 'all'],
                        help="Role(s) to filter by ('gpt', 'system', or 'all'). Default is 'gpt'.")
    parser.add_argument('--min_ngram', type=int, default=3, help="Minimum n-gram length (default is 3).")
    parser.add_argument('--max_ngram', type=int, default=5, help="Maximum n-gram length (default is 5).")
    parser.add_argument('--stopword_limit', type=int, default=1, help="Maximum number of stopwords allowed in n-grams (default is 1).")
    parser.add_argument('--punctuation_limit', type=int, default=1, help="Maximum number of punctuation tokens allowed in n-grams (default is 1).")
    parser.add_argument('--no_punctuation', action='store_true', help="Toggle punctuation filtering. If set, punctuation will be removed from tokens.")

    args = parser.parse_args()

    lines = process_jsonl(args.input, args.role_filter, no_punctuation=args.no_punctuation)

    ngrams = count_ngrams(lines, min_length=args.min_ngram, max_length=args.max_ngram,
                          stopword_limit=args.stopword_limit, punctuation_limit=args.punctuation_limit,
                          no_punctuation=args.no_punctuation)

    print("\nN-gram Counts:")
    for n, counts in ngrams.items():
        print(f"\n{n}-grams:")
        for ngram, count in counts.most_common(10):
            print(f"{' '.join(ngram)}: {count}")
            
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{os.path.basename(args.input)}_ngram_analysis.json")
        
        # Convert Counter objects to dictionaries for JSON serialization
        output_data = {}
        for n, counts in ngrams.items():
            output_data[str(n)] = {' '.join(ngram): count for ngram, count in counts.most_common(50)}
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis saved to {output_file}")

if __name__ == "__main__":
    main()
