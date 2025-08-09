import pandas as pd
from pathlib import Path
import csv
from src.utils.text_processing import get_rule_intervention_indices
def get_swap_dataset(args):
    """Load layer-swap intervention dataset."""
    dataset_folder = Path(args.dataset_dir)
    
    # Rule intervention uses different datasets
    dataset_mapping = {
        "layer_swap_all": 'layer_swap_all.csv',
        "layer_swap_if_then": 'layer_swap_if_then.csv'
    }
    
    if args.rule_type not in dataset_mapping:
        raise ValueError(f"Unknown layer swap type: {args.rule_type}")
    
    dataset_path = dataset_folder / dataset_mapping[args.rule_type]
    
    if args.verbose:
        print(f"Loading rule dataset from {dataset_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    with open(dataset_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    base_inputs = []
    base_labels = []
    
    for i in range(1, len(data)):  # Skip header
        base = data[i][0]
        question = data[i][1]
        base_answer = data[i][2]
        
        base_input = base + " Question: " + question + "?"
        base_inputs.append(base_input)
        base_labels.append(base_answer)
    
    print(f"Loaded rule dataset with {len(base_inputs)} samples")
    return base_inputs, base_labels

def load_swap_data(args, tokenizer):
    """Load and preprocess rule intervention data."""
    base_inputs, base_labels= get_swap_dataset(args)
    
    processed_data = []
    
    print(f"Processing {len(base_inputs)} rule samples...")
    
    for i, (base_input, base_label) in enumerate(
        zip(base_inputs, base_labels)
    ):
        base_tokenized = tokenizer(base_input, padding=False, truncation=False, return_tensors="pt")
        
        processed_data.append({
            'Base_Input': base_input,
            'Base_Label': base_label,
            'Base_Tokenized': base_tokenized,
        })
    
    print(f"Processed {len(processed_data)} valid rule samples")
    return pd.DataFrame(processed_data)