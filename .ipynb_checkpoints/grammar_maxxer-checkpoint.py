import jsonlines
import os
import logging
import language_tool_python
import argparse
from typing import Dict
from tqdm import tqdm

class GrammarMaxxer:
    def __init__(self, input_file: str, toggles: Dict[str, bool]):
        self.input_file = input_file
        self.toggles = toggles
        self.tool = language_tool_python.LanguageTool('en-US')

    def validate_file(self):
        """Validate the selected file."""
        if not self.input_file.endswith('.jsonl'):
            logging.error("Invalid file type. Please select a .jsonl file.")
            return False
        return True

    def prepare_output_file(self):
        """Prepare the output file path."""
        output_dir = os.path.join(os.path.dirname(__file__), "corrected")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        return os.path.join(output_dir, f"{base_name}-corrected.jsonl")

    def process_file(self, output_file, update_corrections_callback):
        """Process the input file and write the corrected text to the output file."""
        # First count the lines
        total_lines = sum(1 for _ in jsonlines.open(self.input_file))
        
        with jsonlines.open(self.input_file) as reader:
            with jsonlines.open(output_file, mode='w') as writer:
                with tqdm(total=total_lines, desc="Processing conversations") as pbar:
                    for conversation in reader:
                        corrected_conversation = self.correct_conversation(conversation, update_corrections_callback)
                        writer.write(corrected_conversation)
                        pbar.update(1)

    def correct_conversation(self, conversation, update_corrections_callback):
        """Correct the text in a conversation and update the live tracker."""
        for turn in conversation.get('conversations', []):
            if turn.get('from') == 'gpt':
                original_text = turn.get('value', '')
                corrected_text = self.correct_text(original_text)
                turn['value'] = corrected_text
                update_corrections_callback(original_text, corrected_text)
        return conversation

    def correct_text(self, text):
        """Correct text using a multi-step process."""
        corrections = {
            'grammar': self.correct_with_grammar
        }
        for key, func in corrections.items():
            if self.toggles[key] == 'on':
                text = func(text)
        return text.strip()

    def correct_with_grammar(self, text):
        """Correct grammar using LanguageTool."""
        matches = self.tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text


def cli_progress_callback(original: str, corrected: str) -> None:
    """Simple progress callback for CLI usage."""
    if original != corrected:
        tqdm.write("Made corrections:")
        tqdm.write(f"Original: {original[:100]}...")
        tqdm.write(f"Corrected: {corrected[:100]}...")
        tqdm.write("-" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Grammar correction tool for JSONL conversation files")
    parser.add_argument("input", help="Input JSONL file to process")
    parser.add_argument("output_dir", help="Directory to save the corrected output")
    parser.add_argument("--disable-grammar", action="store_false", dest="grammar",
                       help="Disable grammar correction")
    
    args = parser.parse_args()
    
    # Convert args to toggles format
    toggles = {
        "grammar": "on" if args.grammar else "off"
    }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output file path
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_file = os.path.join(args.output_dir, f"{base_name}_corrected.jsonl")
    
    # Initialize and run the grammar maxxer
    maxxer = GrammarMaxxer(args.input, toggles)
    
    if not maxxer.validate_file():
        return
    
    print(f"Processing {args.input}")
    print(f"Output will be saved to {output_file}")
    
    maxxer.process_file(output_file, cli_progress_callback)
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
