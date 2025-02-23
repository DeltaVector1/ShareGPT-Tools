import spacy
import jsonlines
import json
import os
import re
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "protectai/distilroberta-base-rejection-v1"

nlp = None
tokenizer = None
model = None
device = 'cpu'

def initialize_models():
    global nlp, tokenizer, model, device
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

def process_file(input_file, threshold, batch_size):
    if not input_file.endswith('.jsonl'):
        raise ValueError("Input file must be a .jsonl file")

    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Threshold must be between 0.0 and 1.0.")

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    output_dir = "classified"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}-classified.jsonl")

    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = [clean_text(line) for line in infile.readlines() if validate_utf8(line)]

    with jsonlines.open(output_file, mode='w') as writer, tqdm(total=len(lines), desc="Processing", unit=" lines") as pbar:
        batch = []
        for line in lines:
            conversation = json.loads(line)
            batch.append(conversation)
            if len(batch) >= batch_size:
                process_batch(batch, threshold, writer)
                batch = []
            pbar.update(1)

        if batch:
            process_batch(batch, threshold, writer)
            pbar.update(len(batch))

    print(f"Processing complete. Output saved to {output_file}")

def process_batch(batch, threshold, writer):
    for conversation in batch:
        result = process_conversation(conversation, threshold)
        if result:
            writer.write(json.dumps(result, ensure_ascii=False) + '\n')

def process_conversation(conversation, threshold):
    sentences = []
    for turn in conversation.get('conversations', []):
        if turn.get('from') == 'gpt':
            value = clean_text(turn.get('value', ''))
            doc = nlp(value)
            sentences.extend(extract_sentences(doc))
    
    classifications = predict(sentences)
    positive_count = sum(1 for classification in classifications if classification['positive'] > threshold)
    
    return conversation if positive_count == 0 else None

def extract_sentences(doc):
    return [clean_text(sent.text.strip()) for sent in doc.sents]

def predict(texts):
    if not texts:
        return []
    
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        
    logits = outputs.logits
    predictions = torch.sigmoid(logits).cpu().numpy()
    
    return [{"positive": float(pred[1]), "negative": float(pred[0])} for pred in predictions]

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text

def validate_utf8(text):
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSONL conversations based on sentiment analysis")
    parser.add_argument("--input", required=True, help="Path to the input JSONL file")
    parser.add_argument("--threshold", type=float, required=True, help="Rejection threshold (0.0 to 1.0)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing (default: 32)")
    
    args = parser.parse_args()
    
    initialize_models()
    process_file(args.input, args.threshold, args.batch_size)