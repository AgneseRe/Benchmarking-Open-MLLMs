# LLaVa Evaluator for Odd-One-Out Anomaly Detection

import torch
import argparse
import pandas as pd
import json, re, sys, time

from PIL import Image
from typing import Tuple
from pathlib import Path

from common import generate_system_instruction, generate_prompt
from common import GenericEvaluator, Config, evaluate_model, extract_odd_index, get_model_load_args

from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig

# ===== ARGUMENT PARSER ===== 
def parse_arguments():

    parser = argparse.ArgumentParser(description='Evaluate LLaVa models')
    
    parser.add_argument('--model', type=str, default='llava-v1.6-mistral-7b-hf', choices = ['llava-v1.6-mistral-7b-hf',
                        'llava-onevision-qwen2-7b-ov-hf', 'llava-v1.6-34b-hf'], help='Model to use for evaluation')
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

# ===== LLAVA EVALUATOR =====
class LLaVaEvaluator(GenericEvaluator):
    
    def __init__(self, config, model_name: str = 'llava-hf/llava-v1.6-mistral-7b-hf'):
        print(f"Loading model: {model_name}...")
        
        self.config = config
        load_kwargs = get_model_load_args(model_name)
    
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        self.model_name = model_name
    
    def predict(self, original_image_path: str, som_image_path: str, class_name: str, use_class_name: bool) -> Tuple[str, str, float]:
        """Query LLaVa. It returns the odd_index, output response and inference time."""
        try:
            system_instruction = generate_system_instruction(self.config)
            prompt = generate_prompt(self.config, class_name, use_class_name)

            messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": system_instruction}
                ]},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{original_image_path}"},
                    {"type": "image", "image": f"file://{som_image_path}"},
                    {"type": "text", "text": prompt},
                ]}
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images = [Image.open(original_image_path).convert('RGB'), Image.open(som_image_path).convert('RGB')]
            inputs = self.processor(text=text, images=images, padding=True, return_tensors="pt").to(self.config.DEVICE)

            # Generation Config (to avoid warning like: The following generation flags are not valid and may be ignored: ['temperature'].)
            generation_config = GenerationConfig(
                do_sample=False,
                max_new_tokens = 1024 if self.config.USE_COT else 64,
                bos_token_id=self.model.config.bos_token_id,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                repetition_penalty=1.05,
                temperature=None,
                top_p=None
            )

            # Inference and time spent
            start_time = time.time()
            generated_ids = self.model.generate(**inputs, generation_config=generation_config) 
            end_time = time.time()
            
            # Model answer and tokens used
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            tokens_used = len(generated_ids_trimmed[0])
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

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

            return (str(odd_index), output_text, end_time - start_time, tokens_used)
        except json.JSONDecodeError:
            print(f"Data deserialized not a valid JSON for '{original_image_path}'.")
            print(f"Response: {output_text}")
            return ("Error", output_text, 0.0, 0)
        except Exception as e:
            print(f"Error for '{original_image_path}': {e}")
            return ("Error", output_text, 0.0, 0)  

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
        llava_evaluator = LLaVaEvaluator(config, f'llava-hf/{args.model}')
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    # Run evaluation
    print("\nStarting evaluation...")
    start_time = time.time()
    
    results_df = evaluate_model(
        llava_evaluator, 
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