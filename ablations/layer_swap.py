
import pickle as pkl
import os
import torch
import numpy as np
import csv
import pickle
from tqdm.autonotebook import tqdm
from pathlib import Path
import gc
from src.data.layer_swap_dataset import load_swap_data
from src.models.model_loader import get_model_and_tokenizer

def run_layer_swap(args, project_root, device):
    """Run rule-based intervention across all layers."""
    model, tokenizer, num_layers = get_model_and_tokenizer(args, device, project_root)
    args.num_layers = num_layers
    processed_df = load_swap_data(args,tokenizer)
    base_inputs = processed_df['Base_Input'].tolist()
    base_labels = processed_df['Base_Label'].tolist()   
    base_tokenized = processed_df['Base_Tokenized'].tolist()
    
    if hasattr(args, 'max_samples') and args.max_samples > 0:
        base_inputs = base_inputs[:args.max_samples]
        base_labels = base_labels[:args.max_samples]
    
    if args.verbose:
        print(f"Running rule intervention on {len(base_inputs)} samples")
    
    replacement_layer_all = []
    layers_output = []
    predicted_labels = []
    probability_drops = []
    mean_layer = {}
    for i in tqdm(range(len(base_inputs)), desc="Processing samples"):
        base_input = base_inputs[i]
        base_label = base_labels[i]
        replacement_layer_prob = []
        if os.path.exists(f"{args.model_name}_sample_{i}.pkl"):
            continue
        result_rep_all, result_rep_layer_prob = process_layer_swap(
            args, model, tokenizer, base_input,base_label,replacement_layer_prob,probability_drops,replacement_layer_all,layers_output,predicted_labels,i
        )
        if args.model_name == 'gpt2':
            layers = {i: [] for i in range(12)}
            for item in range(len(result_rep_all)):
                for layer in range(len(result_rep_all[item][0])):
                    layers[layer].append(result_rep_all[item][0][layer])
            mean_layer = {i: [] for i in range(12)}
            for layer in range(12):
                print(np.array(layers[layer]).shape)
                mean_layer[layer] = np.array(layers[layer]).mean(axis = 0)
        elif args.model_name == 'deepseek':
            LAYERS = 24
            layers = {i: [] for i in range(LAYERS)}
            mean_layer = {i: [] for i in range(LAYERS)}
            for i in range(10):
                with open(f"deepseek_{args.swap_type}_{i}.pkl", "rb") as f:
                    data = pkl.load(f)
                    for layer in range(LAYERS):
                        layers[layer].append(data[layer])
            for layer, items in layers.items():
                mean_layer[layer] = np.array(items).mean(axis = 0)
        elif args.model_name == 'gemma':
            LAYERS = 18
            layers = {i: [] for i in range(LAYERS)}
            mean_layer = {i: [] for i in range(LAYERS)}
            for i in range(10):
                with open(f"gemma_{args.swap_type}_{i}.pkl", "rb") as f:
                    data = pkl.load(f)
                    for layer in range(LAYERS):
                        layers[layer].append(data[layer])
            for layer, items in layers.items():
                mean_layer[layer] = np.array(items).mean(axis = 0)
    return mean_layer

def process_layer_swap(args, model, tokenizer, base_input 
                       ,base_label,replacement_layer_prob,probability_drops,replacement_layer_all,layers_output,predicted_labels,i):
    label_map = {"True": 0, "False": 1, "Unknown": 2}
    base_label_idx = label_map.get(str(base_label))
    with model.trace(input) as tracer:
        original_output = model.score.output.save()

    original_prob = torch.softmax(original_output[0][-1], dim=-1).detach().cpu().numpy()
    predicted_label = "True" if np.argmax(original_prob) == 0 else "False" if np.argmax(original_prob) == 1 else "Unknown"
    assert str(predicted_label) == str(base_label)
    if args.model_name == "gpt2":
        LAYERS = 12
        for replacement_layer in range(LAYERS):
            for layer in range(replacement_layer,LAYERS):
                with model.trace(input) as tracer:
                    original_output = model.score.output.save()
                    attention = model.transformer.h[layer].attn.c_attn.output.save()

                original_prob = torch.softmax(original_output[0][-1], dim=-1).detach().cpu().numpy()
                predicted_label = "True" if np.argmax(original_prob) == 0 else "False" if np.argmax(original_prob) == 1 else "Unknown"
                assert str(predicted_label) == str(base_label_idx)

                with model.trace(input) as tracer:
                    pre_layer = model.transformer.h[replacement_layer].output.save()
                    model.transformer.h[layer].output = pre_layer
                    layer_output = model.transformer.h[layer].output[0].save()
                    output = model.score.output.save()

                layers_output.append(layer_output.cpu())

                output_ = torch.softmax(output, dim=-1).detach().cpu()
                p_labels = "True" if np.argmax(np.array(output_)[:, -1, :]) == 0 else "False" if np.argmax(np.array(output_)[:,-1, :]) == 1 else "Unknown"
                predicted_labels.append(p_labels)
                probabilities = torch.softmax(output[0][-1], dim=-1).detach().cpu().numpy()
                prob_of_correct_class = probabilities[base_label_idx]
                probability_drops.append(1 - (original_prob[base_label_idx] - prob_of_correct_class))

                del original_output, attention, pre_layer, layer_output, output
                torch.cuda.empty_cache()
            replacement_layer_prob.append(probability_drops)
        replacement_layer_all.append([replacement_layer_prob])
    

    elif args.model_name == "deepseek":
        LAYERS = 24
        for replacement_layer in range(LAYERS):
            probability_drops = []
            for layer in range(replacement_layer,LAYERS):
                with model.trace(base_input) as tracer:
                    pre_layer = model.model.layers[replacement_layer].output.save()
                    model.model.layers[layer].output = pre_layer
                    output = model.score.output.save()

                output_ = torch.softmax(output, dim=-1).detach().cpu()
                p_labels = "True" if np.argmax(np.array(output_)[:, -1, :]) == 0 else "False" if np.argmax(np.array(output_)[:,-1, :]) == 1 else "Unknown"
                predicted_labels.append(p_labels)
                probabilities = torch.softmax(output[0][-1], dim=-1).detach().cpu().numpy()
                prob_of_correct_class = probabilities[base_label_idx]
                probability_drops.append(1 - (original_prob[base_label_idx] - prob_of_correct_class))

                del pre_layer, output
                torch.cuda.empty_cache()
            replacement_layer_prob.append(probability_drops)

        with open(f"deepseek_{args.swap_type}_{i}.pkl", "wb") as f:
            pkl.dump(replacement_layer_prob, f)
    elif args.model_name == "gemma":
        LAYERS = 18
        for replacement_layer in range(LAYERS):
            probability_drops = []
            for layer in range(replacement_layer,LAYERS):
                with model.trace(base_input) as tracer:
                    pre_layer = model.model.layers[replacement_layer].output.save()
                    model.model.layers[layer].output = pre_layer
                    output = model.score.output.save()

                output_ = torch.softmax(output, dim=-1).detach().cpu()
                p_labels = "True" if np.argmax(np.array(output_)[:, -1, :]) == 0 else "False" if np.argmax(np.array(output_)[:,-1, :]) == 1 else "Unknown"
                predicted_labels.append(p_labels)
                probabilities = torch.softmax(output[0][-1], dim=-1).detach().cpu().numpy()
                prob_of_correct_class = probabilities[base_label_idx]
                probability_drops.append(1 - (original_prob[base_label_idx] - prob_of_correct_class))

                del pre_layer, output
                torch.cuda.empty_cache()
            replacement_layer_prob.append(probability_drops)

        with open(f"gemma_{args.swap_type}_{i}.pkl", "wb") as f:
            pkl.dump(replacement_layer_prob, f)
    else:
        raise NotImplementedError(f"Model structure not defined for {args.model_name}")    
    return replacement_layer_all,replacement_layer_prob
