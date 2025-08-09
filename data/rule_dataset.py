import pandas as pd
from pathlib import Path
import csv
from src.utils.text_processing import get_rule_intervention_indices
def get_rule_dataset(args):
    """Load rule-based intervention dataset."""
    dataset_folder = Path(args.dataset_dir)
    
    # Rule intervention uses different datasets
    dataset_mapping = {
        "all_rule": 'rule_inference_all.csv',
        "if_then_rule": 'rule_inference.csv'
        # "combined_rule": 'rule_inference_all_new.csv'
    }
    
    if args.rule_type not in dataset_mapping:
        raise ValueError(f"Unknown rule type: {args.rule_type}")
    
    dataset_path = dataset_folder / dataset_mapping[args.rule_type]
    
    if args.verbose:
        print(f"Loading rule dataset from {dataset_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    
    # Load and process the data
    with open(dataset_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    
    source_inputs = []
    base_inputs = []
    base_labels = []
    source_labels = []
    
    for i in range(1, len(data)):  # Skip header
        base = data[i][0]
        source = data[i][1]
        question = data[i][2]
        base_answer = data[i][3]
        source_answer = data[i][4]
        
        source_input = source + " Question: " + question + "?"
        base_input = base + " Question: " + question + "?"
        
        source_inputs.append(source_input)
        base_inputs.append(base_input)
        base_labels.append(base_answer)
        source_labels.append(source_answer)
    
    print(f"Loaded rule dataset with {len(source_inputs)} samples")
    return source_inputs, base_inputs, base_labels, source_labels

def load_rule_data(args, tokenizer):
    """Load and preprocess rule intervention data."""
    
    # Get rule dataset
    source_inputs, base_inputs, base_labels, source_labels = get_rule_dataset(args)
    
    processed_data = []
    
    print(f"Processing {len(source_inputs)} rule samples...")
    
    for i, (source_input, base_input, base_label, source_label) in enumerate(
        zip(source_inputs, base_inputs, base_labels, source_labels)
    ):
        
        # Get rule indices
        source_start_idx, source_end_idx, _ = get_rule_intervention_indices(
            tokenizer, source_input, args.rule_type, args.model_name
        )
        base_start_idx, base_end_idx, _ = get_rule_intervention_indices(
            tokenizer, base_input, args.rule_type, args.model_name
        )
        
        # Skip if indices not found
        if (source_start_idx is None or source_end_idx is None or 
            base_start_idx is None or base_end_idx is None):
            continue
        
        # Tokenize inputs
        base_tokenized = tokenizer(base_input, padding=False, truncation=False, return_tensors="pt")
        source_tokenized = tokenizer(source_input, padding=False, truncation=False, return_tensors="pt")
        
        processed_data.append({
            'Base_Input': base_input,
            'Source_Input': source_input,
            'Base_Label': base_label,
            'Source_Label': source_label,
            'Base_Tokenized': base_tokenized,
            'Source_Tokenized': source_tokenized,
            'Base_End_Idx': base_end_idx,
            'Source_End_Idx': source_end_idx
        })
    
    print(f"Processed {len(processed_data)} valid rule samples")
    return pd.DataFrame(processed_data)