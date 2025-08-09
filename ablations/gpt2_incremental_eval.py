import logging 
import numpy as np
from tqdm import tqdm
import torch
from data.dataset_filter import filter_samples

def run_layer_intervention(args, model, device):
    tokenizer = model.tokenizer
    filtered_data = filter_samples(tokenizer)
    gpt2_last_layer(model, tokenizer, filtered_data, device)
    gpt2_multi_layer_eval(model, tokenizer, filtered_data, device)
def gpt2_last_layer(model, tokenizer, filtered_data,device):
    model.to("cuda")
    
    filtered_data = filter_samples(tokenizer)
    
    for layer in range(12):
        
        correct = 0
        total = 0
        
        for sample in tqdm(filtered_data):
            input_text, label, dot_positions, question_positions = sample
            # print(input_text)
            
            # Prepare the input tensor
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
            # Get the sequence length
            seq_length = input_ids.shape[1]
            
            allowed_tokens = [token_allowed for token_allowed in range(seq_length) if token_allowed%15 == 1]
            # print(allowed_tokens)
            # Create positions to preserve (dot and question mark positions)
            # preserve_positions = set(dot_positions + question_positions)
            # token_posn = list(set(range(seq_length)) - set(dot_positions + question_positions))
            # preserve_positions = set(dot_positions + token_posn)
            # preserve_positions = set([8])
            preserve_positions_ = (allowed_tokens + question_positions) 
            preserve_positions = [pos for pos in preserve_positions_ if pos not in dot_positions]
            with model.trace(input_text) as tracer:
                last_layer_output_ = model.transformer.h[layer].output[0].save()
                
            intervention_mask = torch.ones_like(last_layer_output_.detach())
            for pos in range(seq_length):
                if pos not in preserve_positions:
                    intervention_mask[0, pos, :] = 0.0
            
            # print(intervention_mask)ß
            intervention = last_layer_output_ * intervention_mask
            
            # Get the model's output with intervention
            with model.trace(input_text) as tracer:
                
                # Replace the last layer output with the intervened version
                model.transformer.h[layer].output[0][:] = intervention
                
                # Get the final logits after intervention
                logits = model.score.output.save()
            predicted_class = np.argmax(np.array(logits.detach().cpu().numpy())[0,-1, :])
            
            # Convert label to class index if needed
            if label.lower() in ['true']:
                true_class = 0
            elif label.lower() in ['false']:
                true_class = 1
            else:
                true_class = 2  # Unknown/other

            if predicted_class == true_class:
                correct += 1
            total += 1
        
        final_accuracy = correct / total if total > 0 else 0
        print(f"Layer {layer} accuracy: {final_accuracy:.4f} ({correct}/{total})")
        logging.info(f"Layer {layer} accuracy: {final_accuracy:.4f} ({correct}/{total})")
def gpt2_multi_layer_eval(model, tokenizer, filtered_data,device):
    total_layers=12
    for layer in range(total_layers): 
    
        correct = 0
        total = 0
        
        layer_range = range(total_layers-layer, total_layers)
        
        for sample in tqdm(filtered_data):
            input_text, label, dot_positions, question_positions = sample
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
            seq_length = input_ids.shape[1]
            preserve_positions = set(dot_positions + question_positions)

            with model.trace(input_text) as tracer:
                last_layer_output_ = model.transformer.h[-1].output[0].save()  
                
            intervention_mask = torch.zeros_like(last_layer_output_.detach())
            
            # Zero out all positions except dot and question mark positions
            for pos in range(seq_length):
                if pos not in preserve_positions:
                    intervention_mask[0, pos, :] = 1.0

            
            # First trace: collect outputs from all layers we need to intervene on
            layer_outputs = {}
            
            with model.trace(input_text) as tracer:
                for layer_idx in layer_range:
                    layer_outputs = model.transformer.h[layer_idx].output[0].clone().save()                    
                    layer_outputs = layer_outputs * intervention_mask
                    model.transformer.h[layer_idx].output[0][:] = layer_outputs
                    
                logits = model.score.output.save()
                
            # Get prediction
            predicted_class = np.argmax(np.array(logits.detach().cpu().numpy())[0, -1, :])
            
            # Convert label to class index
            if label.lower() in ['true']:
                true_class = 0
            elif label.lower() in ['false']:
                true_class = 1
            else:
                true_class = 2  # Unknown/other
            
            if predicted_class == true_class:
                correct += 1
            total += 1
        
        final_accuracy = correct / total if total > 0 else 0
        print(f"Layer {layer} accuracy: {final_accuracy:.4f} ({correct}/{total})")
        logging.info(f"Layer {layer} accuracy: {final_accuracy:.4f} ({correct}/{total})")