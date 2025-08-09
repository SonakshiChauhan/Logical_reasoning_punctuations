import re
def find_kth_dot(dot_token_id, prompt_encoded, k):
    """Find the index of the k-th dot token in the encoded prompt."""
    indices = [i for i, n in enumerate(prompt_encoded) if n == dot_token_id]
    
    if len(indices) >= k:
        return indices[k-1]
    else:
        return -1


def find_adjective(text):
    """Extract adjective from text using regex pattern."""
    pattern = r"is\s(?:not)?([^.]+)\."
    match = re.search(pattern, text)

    if match:
        adjective = match.group(1).strip()
    else:
        raise Exception(f"No adjective found in {text}")

    return f" {adjective}"  # prepend space because sensitivity to leading spaces


def find_subject(text):
    """Extract subject from text."""
    words = text.split(" ")
    if text.startswith("The"):
        return " " + words[1]
    return words[0]

def find_all_rule_indices(tokenizer, input_text, model_name):
    """Find indices for 'All X are Y' rule intervention."""
    tokenized_input = tokenizer.tokenize(input_text)
    
    # Different tokenization for different models
    all_token = '▁All' if model_name in ['gemma'] else 'ĠAll'
    
    for i, token in enumerate(tokenized_input):
        if str(token) == all_token:
            all_token_pos = i
            
            # Find '.' after this "All"
            dot_token_pos = None
            for idx in range(all_token_pos, len(tokenized_input)):
                if str(tokenized_input[idx]) == '.':
                    dot_token_pos = idx
                    break
            
            if dot_token_pos is None:
                continue  # Try next "All"
            
            # Extract span and check validity
            span_tokens = tokenized_input[all_token_pos:dot_token_pos+1]
            span_text = tokenizer.convert_tokens_to_string(span_tokens).strip()
            
            # Check restrictions - skip if contains 'and' or ','
            if ('and' in span_text.lower() or ',' in span_text or 
                '_and' in span_text or 'Ġand' in str(span_tokens)):
                continue  # Try next "All"
            
            return all_token_pos, dot_token_pos, span_tokens
    
    return None, None, None

def find_if_then_rule_indices(tokenizer, input_text, model_name):
    """Find indices for 'If X then Y' rule intervention."""
    tokenized_input = tokenizer.tokenize(input_text)
    
    # Different tokenization for different models
    then_token = '▁then' if model_name in ['gemma'] else 'Ġthen'
    
    then_token_pos = None
    for i, token in enumerate(tokenized_input):
        if str(token) == then_token:
            then_token_pos = i
            break
    
    if then_token_pos is None:
        return None, None, None
    
    # Find '.' after 'then'
    dot_token_pos = None
    for idx in range(then_token_pos, len(tokenized_input)):
        if str(tokenized_input[idx]) == '.':
            dot_token_pos = idx
            break
    
    if dot_token_pos is None:
        return None, None, None
    
    return then_token_pos, dot_token_pos, tokenized_input[then_token_pos:dot_token_pos+1]

def get_rule_intervention_indices(tokenizer, input_text, rule_type, model_name):
    """Get intervention indices based on rule type."""
    if rule_type == "all_rule":
        return find_all_rule_indices(tokenizer, input_text, model_name)
    elif rule_type == "if_then_rule":
        return find_if_then_rule_indices(tokenizer, input_text, model_name)
    else:
        # For combined rules, try both
        all_indices = find_all_rule_indices(tokenizer, input_text, model_name)
        if all_indices[0] is not None:
            return all_indices
        return find_if_then_rule_indices(tokenizer, input_text, model_name)