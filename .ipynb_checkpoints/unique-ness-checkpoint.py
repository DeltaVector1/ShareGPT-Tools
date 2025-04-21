import json
import argparse
import re
import os

def analyze_diversity(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    unique_words = len(set(words))
    
    if word_count > 0:
        diversity_score = unique_words / word_count
    else:
        diversity_score = 0
        
    return diversity_score

def process_jsonl(input_dir, output_dir, threshold):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.jsonl'):
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                convo = json.loads(line.strip())
                
                # Calculate average diversity score for all messages in the conversation
                scores = []
                for msg in convo.get('conversations', []):
                    content = msg.get('value', '')
                    score = analyze_diversity(content)
                    scores.append(score)
                
                avg_diversity = sum(scores) / len(scores) if scores else 0
                
                # Only write conversations that meet the threshold
                if avg_diversity >= threshold:
                    outfile.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter ShareGPT JSONL files by word diversity')
    parser.add_argument('input_dir', help='Directory containing input JSONL files')
    parser.add_argument('output_dir', help='Directory to write filtered JSONL files')
    parser.add_argument('--threshold', type=float, default=0.5, help='Minimum diversity score threshold')
    
    args = parser.parse_args()
    process_jsonl(args.input_dir, args.output_dir, args.threshold)