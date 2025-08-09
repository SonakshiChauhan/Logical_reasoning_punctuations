import torch
import numpy as np
import csv
import pickle
from tqdm.autonotebook import tqdm
from pathlib import Path
import gc
import re
# from src.data.rule_dataset import 
from src.data.pre_processing import load_data
from src.data.batching import get_batches, collate_tokenized_data
from src.models.model_loader import get_model_and_tokenizer
from src.utils.file_io import save_to_huggingface
from src.utils.text_processing import get_rule_intervention_indices
from src.data.rule_dataset import get_rule_dataset

def run_rule_intervention(args, project_root, device):
    """Run rule-based intervention across all layers."""
    model, tokenizer, num_layers = get_model_and_tokenizer(args, device, project_root)
    args.num_layers = num_layers
    
    # Load rule-specific data
    source_inputs, base_inputs, base_labels, source_labels = get_rule_dataset(args)
    
    # Limit data for testing if specified
    if hasattr(args, 'max_samples') and args.max_samples > 0:
        source_inputs = source_inputs[:args.max_samples]
        base_inputs = base_inputs[:args.max_samples]
        base_labels = base_labels[:args.max_samples]
        source_labels = source_labels[:args.max_samples]
    
    correct_by_layer = {layer_idx: 0 for layer_idx in range(num_layers)}
    total_by_layer = {layer_idx: 0 for layer_idx in range(num_layers)}
    all_sample_outputs = {layer_idx: [] for layer_idx in range(num_layers)}
    
    if args.verbose:
        print(f"Running rule intervention on {len(source_inputs)} samples")
    
    # Process each sample
    for i in tqdm(range(len(source_inputs)), desc="Processing samples"):
        base_input = base_inputs[i]
        source_input = source_inputs[i]
        base_label = base_labels[i]
        source_label = source_labels[i]
        
        # Get intervention indices
        source_start_idx, source_end_idx, _ = get_rule_intervention_indices(
            tokenizer, source_input, args.rule_type, args.model_name
        )
        base_start_idx, base_end_idx, _ = get_rule_intervention_indices(
            tokenizer, base_input, args.rule_type, args.model_name
        )
        
        # Skip if indices not found
        if (source_start_idx is None or source_end_idx is None or 
            base_start_idx is None or base_end_idx is None):
            if args.verbose:
                print(f"Skipping sample {i}: rule indices not found")
            continue
        
        # Run intervention for each layer
        sample_results = process_rule_sample(
            args, model, tokenizer, base_input, source_input,
            base_end_idx, source_end_idx, base_label, source_label
        )
        
        # Update counts
        for layer_idx, (correct, total) in sample_results.items():
            correct_by_layer[layer_idx] += correct
            total_by_layer[layer_idx] += total
    
    # Calculate accuracies
    layer_accuracies = {}
    for layer_idx in range(num_layers):
        if total_by_layer[layer_idx] > 0:
            layer_accuracies[layer_idx] = correct_by_layer[layer_idx] / total_by_layer[layer_idx]
        else:
            layer_accuracies[layer_idx] = 0.0
    
    if args.verbose:
        print("Rule intervention finished.")
        print(f"Layer accuracies: {layer_accuracies}")
    
    # Save results
    save_rule_results(args, layer_accuracies, correct_by_layer, total_by_layer, num_layers)
    
    return layer_accuracies


def process_rule_sample(args, model, tokenizer, base_input, source_input, 
                       base_end_idx, source_end_idx, base_label, source_label):
    """Process a single sample for rule intervention across all layers."""
    
    # Convert labels to indices
    label_map = {"True": 0, "False": 1, "Unknown": 2}
    source_label_idx = label_map.get(str(source_label))
    
    results = {}
    
    for layer in range(args.num_layers):
        try:
            # Get clean representation from source
            with model.trace(source_input) as tracer:
                if args.model_name == "gpt2":
                    source_hidden_states = model.transformer.h[layer].output[0][:, source_end_idx-1, :].save()
                elif args.model_name in ["gemma", "deepseek"]:
                    source_hidden_states = model.model.layers[layer].output[0][:, source_end_idx-1, :].save()
                else:
                    raise NotImplementedError(f"Model structure not defined for {args.model_name}")
            
            # Intervene on base input
            with model.trace(base_input) as tracer:
                if args.model_name == "gpt2":
                    model.transformer.h[layer].output[0][:, base_end_idx-1, :] = source_hidden_states
                    output_logits = model.score.output.save()
                elif args.model_name in ["gemma", "deepseek"]:
                    model.model.layers[layer].output[0][:, base_end_idx-1, :] = source_hidden_states
                    output_logits = model.score.output.save()
            
            # Get prediction
            predicted_label_idx = torch.argmax(output_logits[-1]).item()
            
            # Check if prediction matches target
            correct = 1 if int(predicted_label_idx) == int(source_label_idx) else 0
            results[layer] = (correct, 1)
            
        except Exception as e:
            if args.verbose:
                print(f"Error in layer {layer}: {e}")
            results[layer] = (0, 1)
    
    return results
def save_rule_results(args, layer_accuracies, correct_by_layer, total_by_layer, num_layers):
    """Save rule intervention results."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV results
    output_file = output_dir / f"rule_intervention_results_{args.rule_type}_{args.model_name}.csv"
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Layer', 'Accuracy', 'Correct', 'Total'])
        for layer_idx in range(num_layers):
            accuracy = layer_accuracies[layer_idx]
            correct = correct_by_layer[layer_idx]
            total = total_by_layer[layer_idx]
            csvwriter.writerow([layer_idx, accuracy, correct, total])
    
    print(f"✅ Rule intervention results saved to {output_file}")