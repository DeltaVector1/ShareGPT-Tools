import os
import argparse
from pathlib import Path
from contextlib import nullcontext, redirect_stderr
from multiprocessing import Pool, cpu_count
import json
import fasttext
import urllib.request
from tqdm import tqdm
import regex as re
import io
import sys

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODEL_FILENAME = "lid.176.ftz"

UNICODE_FILTER_RE = re.compile(
    r'[\u00C0-\u017F\u0400-\u04FF\u0370-\u03FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u4E00-\u9FFF]'
)

_model = None

def download_model_if_missing(model_path):
    if not os.path.exists(model_path):
        print(f"Downloading FastText model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Model downloaded successfully!")

def load_fasttext_model():
    """
    Load the fasttext model while suppressing the warning emitted internally.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    download_model_if_missing(model_path)
    
    # Suppress internal warning by redirecting stderr temporarily
    stderr = sys.stderr
    with io.StringIO() as fake_stderr:
        try:
            sys.stderr = fake_stderr
            model = fasttext.load_model(model_path)
        finally:
            sys.stderr = stderr
    return model

def init_worker():
    global _model
    _model = load_fasttext_model()

def get_model():
    return _model

def is_mostly_english_batch(texts, threshold):
    model = get_model()
    sanitized = [t.replace('\n', ' ') for t in texts]
    predictions = model.predict(sanitized, k=1)
    return [(lang[0], prob[0]) for lang, prob in zip(predictions[0], predictions[1])]

def contains_unwanted_unicode(text):
    return bool(UNICODE_FILTER_RE.search(text))

def extract_gpt_text(conversation):
    return " ".join(
        turn.get("value", "") for turn in conversation.get("conversations", [])
        if isinstance(turn, dict) and turn.get("from") == "gpt" and "value" in turn
    )

def process_batch(args):
    lines, batch_indices, threshold = args
    valid_entries, rejected_entries = [], []
    english_count = non_english_count = json_error_count = 0
    data_objects = []

    for idx, line in zip(batch_indices, lines):
        try:
            data = json.loads(line)
            text = extract_gpt_text(data)
            data_objects.append((idx, line, data, text))
        except Exception:
            rejected_entries.append(line.strip())
            json_error_count += 1

    if not data_objects:
        return valid_entries, rejected_entries, english_count, non_english_count, json_error_count

    predictions = is_mostly_english_batch([x[3] for x in data_objects], threshold)
    for (idx, line, data, text), (lang, prob) in zip(data_objects, predictions):
        if lang == "__label__en" and prob >= threshold and not contains_unwanted_unicode(text):
            valid_entries.append(json.dumps(data, ensure_ascii=False))
            english_count += 1
        else:
            rejected_entries.append(line.strip())
            non_english_count += 1

    return valid_entries, rejected_entries, english_count, non_english_count, json_error_count

def filter_english_jsonl(input_path, output_dir, threshold=0.69, batch_size=256, workers=None, save_rejected=False):
    if workers is None:
        workers = max(1, cpu_count() - 1)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup output paths
    output_path = output_dir / f"{input_path.stem}_filtered.jsonl"
    rejected_path = output_dir / f"{input_path.stem}_rejected.jsonl" if save_rejected else None

    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Filtered output: {output_path}")
    if rejected_path:
        print(f"Rejected output: {rejected_path}")
    print(f"Using {workers} worker processes")
    print(f"Batch size: {batch_size}")
    print(f"English threshold: {threshold}")
    print()

    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    total_lines = len(lines)
    batches = [
        (lines[i:i + batch_size], list(range(i, min(i + batch_size, total_lines))), threshold)
        for i in range(0, total_lines, batch_size)
    ]

    english_total = non_english_total = json_error_total = 0

    with Pool(workers, initializer=init_worker) as pool, \
         open(output_path, 'w', encoding='utf-8') as outfile, \
         open(rejected_path, 'w', encoding='utf-8') if rejected_path else nullcontext() as rejfile:

        with tqdm(total=total_lines, desc="Filtering English JSONL") as pbar:
            for valid_entries, rejected_entries, eng, non_eng, err in pool.imap_unordered(process_batch, batches):
                for entry in valid_entries:
                    outfile.write(entry + "\n")
                if rejfile:
                    for entry in rejected_entries:
                        rejfile.write(entry + "\n")
                english_total += eng
                non_english_total += non_eng
                json_error_total += err
                pbar.update(eng + non_eng + err)

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Total lines processed: {total_lines}")
    print(f"English entries: {english_total} ({english_total/total_lines*100:.1f}%)")
    print(f"Non-English entries: {non_english_total} ({non_english_total/total_lines*100:.1f}%)")
    print(f"JSON errors: {json_error_total} ({json_error_total/total_lines*100:.1f}%)")
    print(f"Output saved to: {output_path}")

    return {
        "total_lines": total_lines,
        "english_total": english_total,
        "non_english_total": non_english_total,
        "json_error_total": json_error_total
    }

def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL files to keep only English conversations using FastText language detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jsonl output/
  %(prog)s data.jsonl results/ --threshold 0.8 --workers 8
  %(prog)s conversations.jsonl filtered/ --save-rejected --batch-size 512
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSONL file to filter"
    )
    
    parser.add_argument(
        "output_dir", 
        type=str,
        help="Directory where filtered output files will be saved"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.69,
        help="Minimum confidence threshold for English detection (default: 0.69)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of lines to process in each batch (default: 256)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)"
    )
    
    parser.add_argument(
        "--save-rejected",
        action="store_true",
        help="Save rejected (non-English) entries to a separate file"
    )

    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print(f"Error: Threshold must be between 0.0 and 1.0, got {args.threshold}")
        sys.exit(1)
    
    # Validate batch size
    if args.batch_size <= 0:
        print(f"Error: Batch size must be positive, got {args.batch_size}")
        sys.exit(1)
    
    # Validate workers
    if args.workers is not None and args.workers <= 0:
        print(f"Error: Number of workers must be positive, got {args.workers}")
        sys.exit(1)

    try:
        filter_english_jsonl(
            input_path=args.input_file,
            output_dir=args.output_dir,
            threshold=args.threshold,
            batch_size=args.batch_size,
            workers=args.workers,
            save_rejected=args.save_rejected
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
