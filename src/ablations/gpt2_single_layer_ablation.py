import numpy as np
import torch
from tqdm import tqdm
import logging

from models.model_loader import get_model_and_tokenizer
from data.dataset_filter import filter_samples

def dot_question_mark_intervention(args, project_root, device):
    model, tokenizer = get_model_and_tokenizer(args, device, project_root)
    model.to("cuda")
    dataset = filter_samples(tokenizer)

    for layer in range(12):
        correct, total = 0, 0

        for input_text, label, dot_pos, q_pos in tqdm(dataset):
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
            seq_len = input_ids.shape[1]

            allowed = [i for i in range(seq_len) if i % 15 == 1]
            preserve = [i for i in (allowed + q_pos) if i not in dot_pos]

            with model.trace(input_text) as tracer:
                output = model.transformer.h[layer].output[0].save()

            mask = torch.ones_like(output.detach())
            for i in range(seq_len):
                if i not in preserve:
                    mask[0, i, :] = 0.0

            intervened = output * mask

            with model.trace(input_text) as tracer:
                model.transformer.h[layer].output[0][:] = intervened
                logits = model.score.output.save()

            pred_class = np.argmax(logits.detach().cpu().numpy()[0, -1, :])
            label = label.lower()
            true_class = 0 if label == "true" else 1 if label == "false" else 2

            if pred_class == true_class:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0
        print(f"Layer {layer} accuracy: {acc:.4f}")
        logging.info(f"Layer {layer} accuracy: {acc:.4f} ({correct}/{total})")

