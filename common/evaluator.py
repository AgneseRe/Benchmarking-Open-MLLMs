# Common utilities for MLLMs evaluation (shared across LLaVa, InternVL, QwenVL models).

import re
import json
import torch
import pandas as pd

from typing import Tuple
from pathlib import Path 
from tqdm.auto import tqdm
from abc import ABC, abstractmethod

from transformers import BitsAndBytesConfig

# ===== CONFIGURATION ===== 
class Config:
    
    def __init__(self, args):
        self.MODEL_ID = args.model
        self.USE_SOM = args.use_som
        self.USE_COT = args.use_cot
        self.USE_CLASS_NAME = args.use_class_name
        
        # Data paths
        self.DATA_DIR = args.data_dir
        self.MARKERS_FILE = args.markers_file
        self.OUTPUT_DIR = args.output_dir
        
        # Performance settings
        self.BATCH_SIZE = args.batch_size
        self.MAX_SAMPLES = args.max_samples
        self.DEVICE = args.device
        
        # Create output directory if needed
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    def print_config(self):
        """Print current configuration."""
        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print(f"Model:           {self.MODEL_ID}")
        print(f"Use SOM:         {self.USE_SOM}")
        print(f"Use CoT:         {self.USE_COT}")
        print(f"Use Class Name:  {self.USE_CLASS_NAME}")
        print(f"Data Directory:  {self.DATA_DIR}")
        print(f"Output Directory: {self.OUTPUT_DIR}")
        print(f"Device:          {self.DEVICE}")
        print(f"Max Samples:     {self.MAX_SAMPLES or 'All'}")
        print("=" * 70)

# ===== PROMPT GENERATION =====
def generate_system_instruction(config) -> str:
    if config.USE_COT:
        instruction = """**First, provide a detailed step-by-step reasoning (Chain-of-Thought) explaining WHY the
        object is the odd one out.** **Second, conclude your response with a single JSON object containing ONLY the
        numeric index of the mark corresponding to the odd-one-out object.**"""
    else:
        instruction = """**Your response MUST be a single JSON object and contain ONLY the numeric index of the mark
        corresponding to the odd-one-out object.** Do not include any other text, explanation, or conversational filler."""
    
    if config.USE_SOM:
        system_instruction = f"""You are an expert visual anomaly detection assistant. Your sole task is to analyze
        the provided images (the original and its 'Set-of-Mark' annotated version) to **identify the odd-one-out object**.
        {instruction} The required JSON structure is: `{{"odd_index": <numeric_index>}}`."""
    else:
        system_instruction = """You are an expert visual anomaly detection assistant. Your sole task is to analyze
        the provided image to **identify the odd-one-out object**. **Your response MUST be a single JSON object that contains
        ONLY the coordinates of the bounding box corresponding to the odd object.** Do not include any other text, explanation, or
        conversational filler. The required JSON structure is: `{"box_2d": [ymin, xmin, ymax, xmax]}`.
        The coordinates **MUST** be an array of four integers: `[ymin, xmin, ymax, xmax]`."""
    
    return ' '.join(system_instruction.split()) # ensure clean system instruction

def generate_prompt(config, class_name: str, use_class_name: bool) -> str:
    som_prompt = [" and its Set-of-Mark annotated version", " marked"] if config.USE_SOM else ["", ""]
    base_prompt = f"Analyze the provided image{som_prompt[0]}. Identify the odd-one-out object among the set of{som_prompt[1]} objects."
    if use_class_name:
        return f"{base_prompt} The objects in the image are instances of the class '{class_name}'."
    else:
        return base_prompt
    
# ===== ABSTRACT EVALUATOR CLASS =====
class GenericEvaluator(ABC):
    @abstractmethod
    def predict(self, original_image_path: str, som_image_path: str, class_name: str, use_class_name: bool) -> Tuple[str, str, float]:
        pass

# ===== MODEL UTILITIES ===== 
def get_model_load_args(model_name: str) :

    is_large_model = any(size in model_name.lower() for size in ["32b", "34b", "38b", "72b", "78b"])
  
    load_kwargs = {
        "attn_implementation": "sdpa",
        "device_map": "auto",
        "dtype": torch.bfloat16
    }

    if is_large_model:
        print("Large model detected. Applying 4-bit quantization.")
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True
        )
        load_kwargs["quantization_config"] = quantization_config
    else:
        print("Small model detected. Loading in full precision (bfloat16).")
    
    return load_kwargs

# ===== ODD INDEX EXTRACTOR =====
def extract_odd_index(text: str) -> int:
    """
    Extract the odd-one-out index from model response.
    
    Returns:
        -1: Refusal detected
        -2: JSON parsing error
        -3: No valid response found
        >=0: Valid index
    """
    clean_text = text.strip().lower()

    # 1. Handle refusal cases (e.g. There are no anomalies detected in the provided images)
    refusal_keywords = ["no anomalies", "no anomaly", "no odd-one-out", "all objects are identical"]
    if any(keyword in clean_text for keyword in refusal_keywords):
        print(f"Refusal detected: {clean_text}")
        return -1

    # 2. parsing json if it is generated
    json_match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
    if json_match:
      try:
        json_response = json.loads(json_match.group(0))
        if isinstance(json_response, dict):
          for key in ['odd_index', 'odd_one_out', 'odd_one', 'index', 'number', 'result', 'target']:
            if key in json_response:
              value = json_response[key]
              if isinstance(value, list) and len(value) > 0:
                value = value[0]
              return int(value)
        elif isinstance(json_response, int):
            return json_response
      except (json.JSONDecodeError, ValueError, TypeError):
        print(f'Format Error in JSON')
        return -2
    
    # 3. no json, but a number in response (V.I. last occurence if CoT)
    numbers = re.findall(r'\d+', clean_text)
    if numbers:
      return int(numbers[-1])

    # 4. generic error
    return -3

# ===== EVALUATION PIPELINE =====
def evaluate_model(evaluator, config, model_name, markers_df, use_class_name: bool = True):

    # Output file setup
    results_file = f'{config.OUTPUT_DIR}/{model_name}_results.csv'
    if Path(results_file).exists():
        results_df = pd.read_csv(results_file, dtype={'predicted_odd_marker': 'Int64'})
        results_list = results_df.to_dict('records')
        processed_images = set(results_df['image_name'].tolist())
        print(f'{len(results_df)} images already processed. Resuming...')
    else:
        results_list = []
        processed_images = set()

    # Update markers_df with unprocessed images
    markers_df['image_stem'] = markers_df['image_path'].apply(lambda x: Path(x).stem)
    working_df = markers_df[~markers_df['image_stem'].isin(processed_images)].copy()

    # Replace prefix in image path for adapting to local environment
    working_df['image_path'] = working_df['image_path'].str.replace('/content', config.DATA_DIR, regex=False)
    working_df['som_image_path'] = working_df['som_image_path'].str.replace('/content', config.DATA_DIR, regex=False)

    # Limit samples if specified
    if config.MAX_SAMPLES:
        working_df = working_df.head(config.MAX_SAMPLES)
        print(f"Limiting evaluation to {config.MAX_SAMPLES} samples")

    # Evaluate each image
    for _, row in tqdm(working_df.iterrows(), total=len(working_df), desc=f"Evaluating with {model_name}"):
        original_image_path = row['image_path']
        som_image_path = row['som_image_path']
        class_name = row['target_type']
        actual_odd_marker = row['odd_marker'] # the actual marker

        try:
            predicted_odd_marker, reasoning, inference_time, tokens_used = evaluator.predict(original_image_path, som_image_path, class_name, use_class_name)

            # Check if prediction is correct
            is_correct = (int(predicted_odd_marker) == actual_odd_marker)

            # Store results
            result_row = {
                'image_name': Path(original_image_path).stem,
                'actual_odd_marker': actual_odd_marker,
                'predicted_odd_marker': predicted_odd_marker,
                'correct': is_correct,
                'target_type': class_name,
                'reasoning': reasoning.replace('\n', ' ').replace('\r', '').strip(),
                'inference_time': inference_time,
                'tokens_used': tokens_used,
            }

            results_list.append(result_row)

            # Save image result to CSV periodically
            if len(results_list) % 10 == 0:
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(results_file, index=False)

            # print(f"Image: {Path(original_image_path).stem}. Actual {actual_odd_marker}, Predicted {predicted_odd_marker}")

        except Exception as e:
            print(f"\nError processing {original_image_path}: {str(e)}")
            break

    print(f"Final save of all {len(results_list)} processed images...")
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(results_file, index=False)

    print(f"\nResults saved to: {results_file}")
    return results_df