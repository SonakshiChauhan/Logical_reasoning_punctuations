import pandas as pd
import torch


def get_batches(args, dataset):
    """Create batches of data with same sequence length."""
    batch_size = args.batch_size
    reference_length = None
    i = 0
    batches = []
    batch = []
    
    while i < len(dataset):
        row = dataset.iloc[i]
        base_tokenized = row['Base_Encoded']
        base_length = base_tokenized['input_ids'].shape[1]

        if reference_length is None:
            reference_length = base_length
            batch.append(row)
        elif base_length == reference_length:
            batch.append(row)
        else:
            batches.append(pd.DataFrame(batch))
            batch = [row]
            reference_length = base_length

        if len(batch) == batch_size:
            batches.append(pd.DataFrame(batch))
            batch = []
            reference_length = None
        i += 1

    if batch:
        batches.append(pd.DataFrame(batch))
    return batches


def collate_tokenized_data(tokenized_list, device):
    """Collate tokenized data into batches."""
    # Extract all 'input_ids' lists/tensors from the list of dicts
    input_ids_list = [item['input_ids'].to(device) for item in tokenized_list]
    # Stack them into a single batch tensor
    input_ids = torch.stack(input_ids_list, dim=0)
    input_ids = input_ids.squeeze(1)

    # Extract all 'attention_mask' lists/tensors
    attention_mask_list = [item['attention_mask'].to(device) for item in tokenized_list]
    # Stack them into a single batch tensor
    attention_mask = torch.stack(attention_mask_list, dim=0)
    attention_mask = attention_mask.squeeze(1)

    # Return the dictionary format the model expects
    return {'input_ids': input_ids, 'attention_mask': attention_mask}