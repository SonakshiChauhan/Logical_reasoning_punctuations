import torch
import numpy as np
import csv
import pickle
from tqdm.autonotebook import tqdm
from pathlib import Path
import gc

from src.data.pre_processing import load_data
from src.data.batching import get_batches, collate_tokenized_data
from src.models.model_loader import get_model_and_tokenizer
from src.utils.file_io import save_to_huggingface


def run_intervention(args, project_root, device):
    """Run intervention across all layers for a given model and intervention type."""
    model, tokenizer, num_layers = get_model_and_tokenizer(args, device, project_root)
    args.num_layers = num_layers
    data = load_data(args, tokenizer)
    batches = get_batches(args, data)

    correct_by_layer = {layer_idx: 0 for layer_idx in range(num_layers)}
    total_by_layer = {layer_idx: 0 for layer_idx in range(num_layers)}
    all_sample_outputs = {layer_idx: [] for layer_idx in range(num_layers)}
    batches = batches[:50]
    
    if args.verbose:
        print(f"Running intervention on {len(batches)} batches with batch size {args.batch_size}.")

    for batch in tqdm(batches, desc="Processing batch...", unit="batch"):
        batch_correct_by_layer, batch_total_by_layer, batch_sample_outputs = process_batch(
            args, batch, model, device
        )
        
        for layer_idx in range(num_layers):
                correct_by_layer[layer_idx] += batch_correct_by_layer[layer_idx]
                total_by_layer[layer_idx] += batch_total_by_layer[layer_idx]
                all_sample_outputs[layer_idx].append(batch_sample_outputs[layer_idx])
        
        torch.cuda.empty_cache()
        gc.collect()

    if args.verbose:
        print("Intervention run finished.")
        print(f"Total samples processed (meeting criteria): {total_by_layer}")
        print(f"Total correct predictions: {correct_by_layer}")

    # Calculate accuracies
    if all(total > 0 for total in total_by_layer.values()):
        layer_accuracies = {layer_idx: correct_by_layer[layer_idx] / total_by_layer[layer_idx] 
                          for layer_idx in range(num_layers)}
    else:
        layer_accuracies = {layer_idx: 0 for layer_idx in range(num_layers)}

    # Save results
    save_results(args, layer_accuracies, correct_by_layer, total_by_layer, all_sample_outputs, num_layers)

    return layer_accuracies

def process_batch(args, batch_df, model,device):
    batch_correct_by_layer = {layer_idx: 0 for layer_idx in range(args.num_layers)}
    batch_total_by_layer = {layer_idx: 0 for layer_idx in range(args.num_layers)}

    batch_sample_outputs = {layer_idx: {} for layer_idx in range(args.num_layers)}
    Base = batch_df['Base'].tolist()
    Source = batch_df['Source'].tolist()
    Question = batch_df['Question'].tolist()
    Base_Answer = batch_df['Base_Answer'].tolist()
    Expected_Answer = batch_df['Expected_Answer'].tolist()
    Base_Prompt = batch_df['Base_Prompt'].tolist()
    Source_Prompt = batch_df['Source_Prompt'].tolist()
    Base_Encoded = batch_df['Base_Encoded'].tolist()
    Source_Encoded = batch_df['Source_Encoded'].tolist()
    Base_Target_String = batch_df['Base_Target_String'].tolist()
    Source_Target_String = batch_df['Source_Target_String'].tolist()
    Base_Target_Idx = batch_df['Base_Target_Idx'].tolist()
    Source_Target_Idx = batch_df['Source_Target_Idx'].tolist()
    Base_Target_Encoded_ID = batch_df['Base_Target_Encoded_ID'].tolist()
    Source_Target_Encoded_ID = batch_df['Source_Target_Encoded_ID'].tolist()
    base_inputs = collate_tokenized_data(Base_Encoded, device)
    source_inputs = collate_tokenized_data(Source_Encoded, device)

    source_token_indices = torch.tensor(Source_Target_Idx, device=device)
    base_token_indices = torch.tensor(Base_Target_Idx, device=device)

    is_sentence_intervention = args.intervention_type in ["two_sentence_on_sentence", "first_sentence_on_sentence"]

    if is_sentence_intervention:
        if not torch.all(base_token_indices == base_token_indices[0]):
            raise ValueError("Base token indices are not the same for all rows in the batch.")
        if not torch.all(source_token_indices == source_token_indices[0]):
            raise ValueError("Source token indices are not the same for all rows in the batch.")
        if not torch.all(base_token_indices == source_token_indices):
            raise ValueError("Base and source token indices do not match for all rows in the batch.")
        slice_end_idx = source_token_indices[0]

    for layer_idx in range(args.num_layers):
        with model.trace() as tracer:
            with tracer.invoke(source_inputs) as invoker:
                if args.model_name == "gpt2":
                    source_hidden_states_layer = model.transformer.h[layer_idx].output[0]
                elif args.model_name in ["gemma", "deepseek"]:
                        source_hidden_states_layer = model.model.layers[layer_idx].output[0]
                else:
                        raise NotImplementedError(f"Model structure not defined for {args.model_name}")
                if is_sentence_intervention:
                    clean_reps = source_hidden_states_layer[:, :slice_end_idx, :].save()
                else:
                    clean_reps = source_hidden_states_layer[:, source_token_indices, :].save()

            with tracer.invoke(base_inputs) as invoker:
                if args.model_name == "gpt2":
                    base_hidden_states_layer = model.transformer.h[layer_idx].output[0]
                    if is_sentence_intervention:
                        base_hidden_states_layer[:, :slice_end_idx, :] = clean_reps
                    else:
                        base_hidden_states_layer[:, base_token_indices, :] = clean_reps

                elif args.model_name in ["gemma", "deepseek"]:
                    base_hidden_states_layer = model.model.layers[layer_idx].output[0]
                    if is_sentence_intervention:
                        base_hidden_states_layer[:, :slice_end_idx, :] = clean_reps
                    else:
                        base_hidden_states_layer[:, base_token_indices, :] = clean_reps
                else:
                    raise NotImplementedError(f"Model structure not defined for {args.model_name}")

                output_logits = model.score.output.save()

        if args.verbose and layer_idx == 1:
            print(f"{args.intervention_type}/{args.model_name} intervention on layer {layer_idx} complete.")
            print(f"Output logits shape: {output_logits.shape}")
        labels = ["True" if np.argmax(np.array(out)[-1, :]) == 0 else
        "False" if np.argmax(np.array(out)[-1, :]) == 1 else
        "Unknown" for out in output_logits.detach().cpu().tolist()]
        current_correct = sum([1 for label, expected in zip(labels, Expected_Answer) if label == expected])
        batch_correct_by_layer[layer_idx] += current_correct
        batch_total_by_layer[layer_idx] += len(labels)
        batch_sample_outputs[layer_idx] = {
            'logits': output_logits.detach().cpu(),
            'predictions': labels,
            'expected_answers': Expected_Answer,
            'base_prompts': Base_Prompt,
            'source_prompts': Source_Prompt
        }

    return batch_correct_by_layer, batch_total_by_layer,batch_sample_outputs

import csv
import pickle
from pathlib import Path
from src.utils.file_io import save_to_huggingface

def save_results(args, layer_accuracies, correct_by_layer, total_by_layer, all_sample_outputs, num_layers):
    """Save intervention results to CSV and PKL files."""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save CSV results (layer-wise summary)
    output_file = output_dir / f"intervention_results_{args.intervention_type}_{args.model_name}.csv"
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Layer', 'Accuracy', 'Correct', 'Total'])
        for layer_idx in range(num_layers):
            accuracy = layer_accuracies[layer_idx]
            correct = correct_by_layer[layer_idx]
            total = total_by_layer[layer_idx]
            csvwriter.writerow([layer_idx, accuracy, correct, total])
    print(f"Results saved to {output_file}")

    # 2. Save detailed results (per-sample, per-layer data)
    detailed_output_file = output_dir / f"detailed_results_{args.intervention_type}_{args.model_name}.pkl"
    with open(detailed_output_file, 'wb') as f:
        pickle.dump(all_sample_outputs, f)
    print(f"Detailed results saved to {detailed_output_file}")

