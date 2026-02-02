# MiniCPM Evaluator for Odd-One-Out Anomaly Detection

import torch
import argparse
import pandas as pd
import json, re, sys, time

from PIL import Image
from typing import Tuple
from pathlib import Path

from common import generate_system_instruction, generate_prompt
from common import GenericEvaluator, Config, evaluate_model, extract_odd_index, get_model_load_args

from transformers import AutoProcessor, AutoModelForCausalLM

# ===== ARGUMENT PARSER ===== 
def parse_arguments():

    parser = argparse.ArgumentParser(description='Evaluate MiniCPM models')
    
    parser.add_argument('--model', type=str, default='MiniCPM-V-4', choices = ['MiniCPM-V-4'], 
                        help='Model to use for evaluation') # for scalability and future comparison with other models
    parser.add_argument('--som', dest='use_som', action='store_true', default=False, help='Use of Set-of-Mark')
    parser.add_argument('--cot', dest='use_cot', action='store_true', default=False, help='Use of Chain-of-Thought')
    parser.add_argument('--class-name', dest='use_class_name', action='store_true', default=False, help='Include class name in prompt')
    
    parser.add_argument('--data-dir', type=str, default=None, help='Directory containing SOM data') # in BeeGFS
    parser.add_argument('--markers-file', type=str, default='O3_output/som/odd_markers.csv', help='Markers CSV filename')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()

# ===== MINICPM EVALUATOR =====
class MiniCPMEvaluator(GenericEvaluator):
    
    def __init__(self, config, model_name: str = 'openbmb/MiniCPM-V-4'):
        print(f"Loading model: {model_name}...")
        
        self.config = config
        load_kwargs = get_model_load_args(model_name)
    
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        self.model_name = model_name
    
    def predict(self, original_image_path: str, som_image_path: str, class_name: str, use_class_name: bool) -> Tuple[str, str, float]:
        """Query MiniCPM. It returns the odd_index, output response and inference time."""
        output_text = ""
        try:
            system_instruction = generate_system_instruction(self.config)
            prompt = generate_prompt(self.config, class_name, use_class_name)

            original_image = Image.open(original_image_path).convert('RGB')
            som_image = Image.open(som_image_path).convert('RGB')

            messages = [
                {
                    "role": "user", 
                    "content": [original_image, som_image, prompt]
                }
            ]

            # Inference and time spent
            # https://huggingface.co/openbmb/MiniCPM-V-4/blob/main/modeling_minicpmv.py
            start_time = time.time()
            output_text = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.processor.tokenizer,
                max_new_tokens=1024 if self.config.USE_COT else 64,
                sampling=False, # greedy search
                system_prompt = system_instruction, # 'role': 'system'
                repetition_penalty=1.05,
                num_beams=1,
            )
            end_time = time.time()

            # Clean potential markdown formatting
            output_text = output_text.strip()
            if output_text.lower().startswith("```json"):
                output_text = output_text[7:].strip()
            if output_text.lower().startswith("```"):
                output_text = output_text[3:].strip()
            if output_text.lower().endswith("```"):
                output_text = output_text[:-3].strip()

            # get odd index from JSON
            odd_index = extract_odd_index(output_text)

            tokens_used = len(self.processor.tokenizer.encode(output_text))

            return (str(odd_index), output_text, end_time - start_time, tokens_used)
        except json.JSONDecodeError:
            print(f"Data deserialized not a valid JSON for '{original_image_path}'.")
            print(f"Response: {output_text}")
            return ("Error", output_text, 0.0, 0)
        except Exception as e:
            print(f"Error for '{original_image_path}': {e}")
            return ("Error", str(e), 0.0, 0)  

# ===== MAIN EXECUTION =====
def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    config = Config(args)
    config.print_config()
    
    # Verify CUDA availability if requested
    if config.DEVICE == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        config.DEVICE = 'cpu'
    
    if config.DEVICE == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load markers dataset
    markers_path = f'{config.DATA_DIR}/{config.MARKERS_FILE}'
    
    if not Path(markers_path).exists():
        print(f"ERROR: Markers file not found at {markers_path}")
        sys.exit(1)
    
    print(f"Loading markers from: {markers_path}")
    markers_df = pd.read_csv(markers_path)
    print(f"Loaded {len(markers_df)} images from markers dataset")
    
    # Initialize evaluator
    try:
        minicpm_evaluator = MiniCPMEvaluator(config, f'openbmb/{args.model}')
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    # Run evaluation
    print("\nStarting evaluation...")
    start_time = time.time()
    
    results_df = evaluate_model(
        minicpm_evaluator, 
        config,
        args.model.lower(), 
        markers_df, 
        use_class_name=config.USE_CLASS_NAME
    )
    
    end_time = time.time()
    
    # Final statistics
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total samples:     {len(results_df)}")
    print(f"Correct:           {results_df['correct'].sum()}")
    print(f"Accuracy:          {results_df['correct'].mean():.2%}")
    print(f"Total time:        {end_time - start_time:.2f}s")
    print(f"Avg time/sample:   {(end_time - start_time) / len(results_df):.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()