import logging
import os
import torch
import json
import spacy
import asyncio
import aiofiles
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CharacterSlopFilter:
    def __init__(self, model_name="kubernetes-bad/character-slop-classifier", batch_size=58, confidence_margin=0.1):
        self.classifier = pipeline("text-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        self.batch_size = batch_size
        self.confidence_margin = confidence_margin
        self.classification_cache = defaultdict(dict)

        logging.info(f"Pipeline loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def split_into_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def classify_sentences(self, sentences):
        to_classify = [sent for sent in sentences if sent not in self.classification_cache]
        if to_classify:
            results = self.classifier(
                to_classify,
                batch_size=self.batch_size,
                truncation=True,
                max_length=512
            )
            for sentence, result in zip(to_classify, results):
                self.classification_cache[sentence] = {"label": result["label"], "score": result["score"]}

        return [self.classification_cache[sent] for sent in sentences]

    async def filter_conversations(self, filepath, output_filepath):
        filtered_conversations = []
        try:
            async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
                lines = await f.readlines()

            for line in tqdm(lines, desc="Processing conversations"):
                try:
                    data = json.loads(line.strip())
                    conversations = data.get("conversations", [])
                    gpt_sentences = []

                    for conversation in conversations:
                        if conversation.get("from") == "gpt":
                            text = conversation.get("value", "")
                            if text:
                                sentences = self.split_into_sentences(text)
                                gpt_sentences.extend(sentences)

                    if gpt_sentences:
                        sentence_results = self.classify_sentences(gpt_sentences)
                        positive_count = sum(
                            1 for result in sentence_results if result["label"] == "positive" and result["score"] > 0.5 + self.confidence_margin
                        )
                        positive_ratio = positive_count / len(gpt_sentences)

                        logging.debug(f"Positive ratio: {positive_ratio} for conversation: {data}")

                        if positive_ratio <= 0.55:
                            filtered_conversations.append(data)

                except json.JSONDecodeError as jde:
                    logging.error(f"JSON decode error in line: {line}, error: {jde}")
                except UnicodeDecodeError as ude:
                    logging.error(f"Unicode decode error in line: {line}, error: {ude}")
                except Exception as e:
                    logging.error(f"Error processing file {filepath}: {e}")

            async with aiofiles.open(output_filepath, "w", encoding="utf-8") as f_out:
                for conversation in filtered_conversations:
                    await f_out.write(json.dumps(conversation, ensure_ascii=False) + "\n")

            logging.info(f"Filtered conversations saved to {output_filepath}")
        except Exception as e:
            logging.error(f"Error processing file {filepath}: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Filter JSONL conversations using a text classifier.")
    parser.add_argument("input", help="Path to the input JSONL file")
    parser.add_argument("output_dir", help="Directory to save the filtered output")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for classification")
    parser.add_argument("--confidence_margin", type=float, default=0.1, help="Confidence margin for classification")

    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output file path
    output_file = os.path.join(args.output_dir, os.path.basename(args.input).replace('.jsonl', '_filtered.jsonl'))
    
    slop_filter = CharacterSlopFilter(batch_size=args.batch_size, confidence_margin=args.confidence_margin)
    await slop_filter.filter_conversations(args.input, output_file)

if __name__ == "__main__":
    asyncio.run(main())
