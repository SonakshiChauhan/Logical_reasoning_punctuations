import pandas as pd
import torch
import sys
import re
from tqdm.autonotebook import tqdm
from src.data.dataset import get_dataset
from src.utils.text_processing import find_kth_dot, find_adjective, find_subject


def load_data(args, tokenizer):
    """Load and preprocess data for intervention experiments."""
    df = get_dataset(args)

    processed_data = []
    try:
        dot_token_id = tokenizer.encode(".", add_special_tokens=False)[0]
        if args.verbose:
             print(f"Using dot token ID: {dot_token_id} ('{tokenizer.decode([dot_token_id])}') for model {args.model_name}")
    except IndexError:
         print(f"ERROR: Could not encode '.' for tokenizer {args.model_name}. Exiting.")
         sys.exit(1)

    print(f"Processing {len(df)} rows from dataset...")
    for index, row_series in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
        row = row_series.to_dict()
        base = str(row.get('Base', ''))
        source = str(row.get('Source', ''))
        question = str(row.get('Question', ''))
        base_answer = str(row.get('Base_Answer', ''))
        expected_answer = str(row.get('Expected_Answer', ''))

        if not all([base, source, question]):
             print(f"Warning: Skipping row {index} due to missing Base, Source, or Question.")
             continue

        # --- Filtering ---
        if "not" in question.lower():
            continue

        # --- Construct Prompts ---
        base_prompt = f"{base} Question: {question}?"
        source_prompt = f"{source} Question: {question}?"

        # --- Tokenize Prompts ---
        base_inputs = tokenizer(base_prompt, padding=False, truncation=False, return_tensors="pt")
        source_inputs = tokenizer(source_prompt, padding=False, truncation=False, return_tensors="pt")

        # --- Get ID lists (remove batch dimension) ---
        if base_inputs['input_ids'].shape[0] != 1 or source_inputs['input_ids'].shape[0] != 1:
            print(f"Warning: Skipping row {index} due to unexpected tokenizer output shape.")
            continue
        base_ids_list = base_inputs['input_ids'][0].tolist()
        source_ids_list = source_inputs['input_ids'][0].tolist()

        if len(base_ids_list) != len(source_ids_list):
            continue

        # --- Process target information based on intervention type ---
        target_info = process_intervention_targets(
            args, base, source, base_ids_list, source_ids_list, 
            dot_token_id, tokenizer, index
        )
        
        if target_info is None:
            continue

        processed_data.append({
            'Base': base,
            'Source': source,
            'Question': question,
            'Base_Answer': base_answer,
            'Expected_Answer': expected_answer,
            'Base_Prompt': base_prompt,
            'Source_Prompt': source_prompt,
            'Base_Encoded': base_inputs,
            'Source_Encoded': source_inputs,
            **target_info
        })

    # --- Create Final DataFrame ---
    if not processed_data:
         print("WARNING: No data rows remained after processing and filtering.")
         return pd.DataFrame()

    filtered_df = pd.DataFrame(processed_data)

    filtered_df['base_len'] = filtered_df['Base_Encoded'].apply(lambda x: x['input_ids'].shape[1])
    filtered_df.sort_values(by=['base_len'], ascending=True, inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)

    if args.verbose:
        print(f"Filtered dataset to {len(filtered_df)} rows.")
        if not filtered_df.empty:
            print("Sample processed row (first after sorting):")
            print(filtered_df.iloc[0][['Base_Prompt', 'Source_Prompt', 'Base_Target_String', 'Base_Target_Idx', 'Base_Target_Encoded_ID']])

    return filtered_df


def process_intervention_targets(args, base, source, base_ids_list, source_ids_list, 
                               dot_token_id, tokenizer, index):
    """Process target information based on intervention type."""
    base_target_string = None
    source_target_string = None
    base_target_idx = -1
    source_target_idx = -1
    base_target_encoded_id = -1
    source_target_encoded_id = -1
    valid_row = False

    try:
        if args.intervention_type == "two_sentence_on_dot":
            base_target_idx = find_kth_dot(dot_token_id, base_ids_list, 2)
            source_target_idx = find_kth_dot(dot_token_id, source_ids_list, 2)
            if base_target_idx != -1 and source_target_idx != -1:
                base_target_string = "."
                source_target_string = "."
                base_target_encoded_id = dot_token_id
                source_target_encoded_id = dot_token_id
                valid_row = True
                
        elif args.intervention_type == "two_sentence_on_sentence":
            base_target_idx = find_kth_dot(dot_token_id, base_ids_list, 2)
            source_target_idx = find_kth_dot(dot_token_id, source_ids_list, 2)
            if (base_target_idx == source_target_idx) and (base_target_idx != -1) and (source_target_idx != -1):
                base_target_string = "."
                source_target_string = "."
                base_target_encoded_id = dot_token_id
                source_target_encoded_id = dot_token_id
                valid_row = True
                
        elif args.intervention_type == "first_sentence_on_dot":
            base_target_idx = find_kth_dot(dot_token_id, base_ids_list, 1)
            source_target_idx = find_kth_dot(dot_token_id, source_ids_list, 1)
            if base_target_idx != -1 and source_target_idx != -1:
                base_target_string = "."
                source_target_string = "."
                base_target_encoded_id = dot_token_id
                source_target_encoded_id = dot_token_id
                valid_row = True
                
        elif args.intervention_type == "first_sentence_on_sentence":
            base_target_idx = find_kth_dot(dot_token_id, base_ids_list, 1)
            source_target_idx = find_kth_dot(dot_token_id, source_ids_list, 1)
            if (base_target_idx == source_target_idx) and (base_target_idx != -1) and (source_target_idx != -1):
                base_target_string = "."
                source_target_string = "."
                base_target_encoded_id = dot_token_id
                source_target_encoded_id = dot_token_id
                valid_row = True
                
        elif args.intervention_type == "subject":
            base_target_string = find_subject(base)
            source_target_string = find_subject(source)
            
        elif args.intervention_type == "adjective":
            base_target_string = find_adjective(base)
            source_target_string = find_adjective(source)
            
        else:
            raise ValueError(f"Unknown intervention type: {args.intervention_type}")

        # Handle subject and adjective interventions
        if args.intervention_type in ["subject", "adjective"]:
            if base_target_string and source_target_string:
                try:
                    base_target_encoded = tokenizer(base_target_string, padding=False, truncation=False, return_tensors="pt")['input_ids']
                    source_target_encoded = tokenizer(source_target_string, padding=False, truncation=False, return_tensors="pt")['input_ids']
                    
                    if base_target_encoded.shape[1] == 2: 
                        base_target_encoded = torch.tensor([[base_target_encoded[0][1]]])
                    if source_target_encoded.shape[1] == 2: 
                        source_target_encoded = torch.tensor([[source_target_encoded[0][1]]])
                except Exception as e:
                    return None

                # Check if it's a single token
                if base_target_encoded.shape[1] == 1 and source_target_encoded.shape[1] == 1:
                    base_target_encoded_id = base_target_encoded[0, 0].item()
                    source_target_encoded_id = source_target_encoded[0, 0].item()
                    try:
                        base_target_idx = base_ids_list.index(base_target_encoded_id)
                        source_target_idx = source_ids_list.index(source_target_encoded_id)
                        valid_row = True
                    except ValueError as e:
                        print(f"Couldn't find object")
                        valid_row = False
                else:
                    valid_row = False

    except Exception as e:
        print(f"Error processing row {index} during type matching or index finding: {e}")
        valid_row = False

    if not valid_row:
        return None

    return {
        'Base_Target_String': base_target_string,
        'Source_Target_String': source_target_string,
        'Base_Target_Idx': base_target_idx,
        'Source_Target_Idx': source_target_idx,
        'Base_Target_Encoded_ID': base_target_encoded_id,
        'Source_Target_Encoded_ID': source_target_encoded_id,
    }

def extract_all_dot_and_question_positions(sample, tokenizer):
    encoded_sample = tokenizer.encode(sample, add_special_tokens=False)
    dot_token_id = tokenizer.encode(".", add_special_tokens=False)[0]
    question_token_id = tokenizer.encode("?", add_special_tokens=False)[0]

    dot_positions = [i for i, token_id in enumerate(encoded_sample) if token_id == dot_token_id]
    question_positions = [i for i, token_id in enumerate(encoded_sample) if token_id == question_token_id]
    return dot_positions, question_positions


