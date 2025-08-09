import torch
import csv
import os
from tqdm import tqdm
from src.models.model_loader import get_model_and_tokenizer
from src.data.dataset_filter import filter_samples


def save_results_and_summary(results_data, output_path, model_name):
    results_by_case = {}
    for result in results_data:
        case = result['case']
        if case not in results_by_case:
            results_by_case[case] = []
        results_by_case[case].append(result)

    for case, case_results in results_by_case.items():
        # Create filename with case and model
        results_file = os.path.join(output_path, f"{case}_{model_name}_results.csv")
        summary_file = os.path.join(output_path, f"{case}_{model_name}_layer_summary.csv")
        
        # Save detailed results for this case
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['layer', 'sample_id', 'input_text', 'true_label', 'predicted_label', 'is_correct'])
            
            for result in case_results:
                writer.writerow([
                    result['layer'], result['sample_id'], 
                    result['input_text'], result['true_label'], 
                    result['predicted_label'], result['is_correct']
                ])

        with open(summary_file, 'w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['layer', 'accuracy', 'correct', 'total'])

            stats = {}
            for result in case_results:
                layer = result['layer']
                correct = result['is_correct']
                stats.setdefault(layer, {"correct": 0, "total": 0})
                stats[layer]["correct"] += int(correct)
                stats[layer]["total"] += 1

            for layer in sorted(stats):
                acc = stats[layer]["correct"] / stats[layer]["total"]
                writer.writerow([layer, acc, stats[layer]["correct"], stats[layer]["total"]])
        
        print(f"Saved results for {case} with {model_name}: {results_file}")
        print(f"Saved summary for {case} with {model_name}: {summary_file}")


def run_necessity_sufficiency(args, project_root, device):
    model, data, tokenizer, _ = get_model_and_tokenizer(args.model_path, args.output_path, args.model_name)
    filtered_data = filter_samples(tokenizer)

    results_data = []

    cases = [
        {"mask_type": "zeros", "preserve": "q_pos", "description": "zeros_q_pos"},
        {"mask_type": "zeros", "preserve": "dot_pos", "description": "zeros_dot_pos"},
        {"mask_type": "zeros", "preserve": "q_dot_pos", "description": "zeros_q_dot_pos"},
        {"mask_type": "ones", "preserve": "q_pos", "description": "ones_q_pos"},
        {"mask_type": "ones", "preserve": "dot_pos", "description": "ones_dot_pos"},
        {"mask_type": "ones", "preserve": "q_dot_pos", "description": "ones_q_dot_pos"}
    ]

    for case_info in cases:
        case_name = case_info["description"]
        print(f"\nRunning case: {case_name}")
        
        for layer in range(18):
            correct = total = 0
            for sample_id, (text, label, dot_pos, q_pos) in enumerate(tqdm(filtered_data, desc=f"Case {case_name} - Layer {layer}")):
                input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
                seq_len = input_ids.shape[1]
                if case_info["preserve"] == "q_pos":
                    preserve_pos = q_pos
                elif case_info["preserve"] == "dot_pos":
                    preserve_pos = dot_pos
                else:  # q_dot_pos
                    preserve_pos = q_pos + dot_pos

                with model.trace(text) as tracer:
                    if args.model_name == 'gpt2':
                        layer_out = model.transformer.h[layer].output[0].save()
                    elif args.model_name in['deepseek', 'gemma']:
                        layer_out = model.model.layers[layer].output[0].save()
                if case_info["mask_type"] == "zeros":
                    mask = torch.zeros_like(layer_out.detach())
                    mask_value = 1.0
                else:  # ones
                    mask = torch.ones_like(layer_out.detach())
                    mask_value = 0.0
                for pos in range(seq_len):
                    if pos not in preserve_pos:
                        mask[0, pos, :] = mask_value

                intervened = layer_out * mask

                with model.trace(text):
                    if args.model_name == 'gpt2':
                        model.transformer.h[layer].output[0][:] = intervened
                    elif args.model_name in ['deepseek', 'gemma']:
                        model.model.layers[layer].output[0][:] = intervened
                    logits = model.score.output.save()

                pred = torch.argmax(logits[0, -1, :]).item()
                true_class = 0 if label.lower() == 'true' else 1 if label.lower() == 'false' else 2
                is_correct = pred == true_class

                if is_correct:
                    correct += 1
                total += 1

                # Store result data
                results_data.append({
                    'case': case_name,
                    'layer': layer,
                    'sample_id': sample_id,
                    'input_text': text,
                    'true_label': true_class,
                    'predicted_label': pred,
                    'is_correct': is_correct
                })

            acc = correct / total if total else 0
            print(f"Case {case_name} - Layer {layer} accuracy: {acc:.4f} ({correct}/{total})")

    # Save all results using the separate function
    save_results_and_summary(results_data, args.output_path, args.model_name)