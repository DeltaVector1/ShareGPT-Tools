# ShareGPT Tools

A comprehensive toolkit for processing, analyzing, and filtering ShareGPT conversation datasets. This collection of tools helps you clean, deduplicate, analyze, and improve the quality of conversation datasets.

## Features

### Dataset Processing
- **Dataset Converter** (`dataset_converter.py`): Converts various JSON/JSONL dataset formats into a standardized conversation format
- **Parquet Converter** (`parquet.py`): Converts Parquet files to JSONL format
- **Dataset Filter** (`dataset_filter.py`): Filters conversations based on various quality criteria

### Quality Improvement
- **Grammar Maxxer** (`grammar_maxxer.py`): Improves grammar in conversations using LanguageTool
- **Deduplication** (`deduplication.py`): Removes duplicate conversations using SHA-256 or MinHash algorithms
- **Binary Classification** (`binary_classification.py`): Filters conversations based on sentiment analysis
- **DeSlopTool** (`DeslopTool.py`, `DeslopTool_classifier.py`): Removes low-quality or "slop" content from conversations

### Analysis
- **N-gram Analyzer** (`ngram_analyzer.py`): Analyzes n-gram patterns in conversations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sharegpt-tools.git
cd sharegpt-tools
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Dataset Conversion

Convert JSON/JSONL datasets to standardized format:
```bash
python dataset_converter.py --input-files input.json --output-dir output/
```

Convert Parquet to JSONL:
```bash
python parquet.py --input data.parquet --output data.jsonl
```

### Quality Improvement

Run grammar improvements:
```bash
python grammar_maxxer.py input.jsonl
```

Deduplicate conversations:
```bash
python deduplication.py --input input.jsonl --out-file output.jsonl --method sha256
```

Filter conversations using binary classification:
```bash
python binary_classification.py --input input.jsonl --threshold 0.8 --batch_size 32
```

Remove low-quality content:
```bash
python DeslopTool.py --dataset input.jsonl --filters filter_criteria.txt
```

### Analysis

Analyze n-grams in conversations:
```bash
python ngram_analyzer.py input.jsonl --role_filter gpt --min_ngram 3 --max_ngram 5
```

## Output Structure

Most tools output processed files in JSONL format with the following structure:
```json
{
  "conversations": [
    {
      "from": "system",
      "value": "System message here"
    },
    {
      "from": "human",
      "value": "User message here"
    },
    {
      "from": "gpt",
      "value": "Assistant response here"
    }
  ]
}
```

## Tool Descriptions

### Dataset Converter
Converts various JSON and JSONL dataset formats into a standardized conversation format. Handles different input structures and normalizes them to a consistent output format.

### Parquet Converter
Converts Parquet files to JSONL format, maintaining data integrity and handling various data types through custom NumPy encoding.

### Dataset Filter
Filters conversations based on criteria such as:
- Blank turns
- Invalid message endings
- Null GPT responses
- Duplicate system messages

### Grammar Maxxer
Uses LanguageTool to improve grammar in conversations, focusing on GPT responses. Supports customizable correction rules.

### Deduplication
Offers two deduplication methods:
- SHA-256: For exact matches
- MinHash: For near-duplicate detection with configurable similarity threshold

### Binary Classification
Uses the DistilRoBERTa model to classify and filter conversations based on sentiment analysis, with configurable threshold and batch processing.

### DeSlopTool
Removes low-quality content using:
- Text-based filtering with custom criteria
- ML-based classification for character-level analysis
- Configurable thresholds and filtering rules

### N-gram Analyzer
Analyzes conversation patterns by:
- Extracting n-grams of configurable length
- Filtering by role (system, human, gpt)
- Handling stopwords and punctuation
- Providing frequency analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your License Here]
