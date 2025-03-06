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
device = 'cuda'

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
    
    # Count total lines for progress bar
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with jsonlines.open(input_file, mode='r') as reader, \
         jsonlines.open(output_file, mode='w') as writer, \
         tqdm(total=total_lines, desc="Processing", unit=" conversations") as pbar:
        
        batch = []
        for conversation in reader:
            if validate_json(conversation):
                batch.append(conversation)
                if len(batch) >= batch_size:
                    process_batch(batch, threshold, writer)
                    pbar.update(len(batch))
                    batch = []
        
        # Process remaining items
        if batch:
            process_batch(batch, threshold, writer)
            pbar.update(len(batch))
    
    print(f"Processing complete. Output saved to {output_file}")

def validate_json(obj):
    # Simple validation to ensure we have a conversation object
    if not isinstance(obj, dict) or 'conversations' not in obj:
        return False
    return True

def process_batch(batch, threshold, writer):
    all_sentences = []
    sentence_indices = []
    conversation_map = []
    
    # Extract all sentences from all conversations into a single batch
    for i, conversation in enumerate(batch):
        sentences = []
        for turn in conversation.get('conversations', []):
            if turn.get('from') == 'gpt':
                value = clean_text(turn.get('value', ''))
                doc = nlp(value)
                extracted = extract_sentences(doc)
                sentences.extend(extracted)
        
        if sentences:
            start_idx = len(all_sentences)
            all_sentences.extend(sentences)
            end_idx = len(all_sentences)
            sentence_indices.append((start_idx, end_idx))
        else:
            # No sentences to process, mark as "keep"
            sentence_indices.append(None)
        
        conversation_map.append(conversation)
    
    # Batch process all sentences
    if all_sentences:
        classifications = predict(all_sentences)
    else:
        classifications = []
    
    # Now determine which conversations to keep
    for i, indices in enumerate(sentence_indices):
        if indices is None:
            # No sentences to check, keep the conversation
            writer.write(conversation_map[i])
            continue
            
        start_idx, end_idx = indices
        conversation_classifications = classifications[start_idx:end_idx]
        
        if not any(c['positive'] > threshold for c in conversation_classifications):
            writer.write(conversation_map[i])

def extract_sentences(doc):
    return [clean_text(sent.text.strip()) for sent in doc.sents if sent.text.strip()]

def predict(texts):
    if not texts:
        return []
    
    # Process in sub-batches if needed for very large batches
    batch_size = 128
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        
        logits = outputs.logits
        predictions = torch.sigmoid(logits).cpu().numpy()
        
        batch_results = [{"positive": float(pred[1]), "negative": float(pred[0])} for pred in predictions]
        results.extend(batch_results)
    
    return results

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text

def validate_utf8(text):
    try:
        if isinstance(text, str):
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