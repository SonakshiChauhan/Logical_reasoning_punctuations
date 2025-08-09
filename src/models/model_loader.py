from nnsight import LanguageModel
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from huggingface_hub import login
from pathlib import Path


def get_model_and_tokenizer(args, device, project_root, num_labels=3):
    """Load model and tokenizer based on name."""
    assert num_labels == 3, "Only 3 labels are supported for now."
    assert args.model_name in ['gpt2', 'gemma', 'deepseek'], "Model name not recognized. Use 'gpt2', 'gemma', or 'deepseek'."
 
    # Handle model path - convert to absolute string path
    if args.model_path:
        model_path = str(Path(args.model_path).resolve())
    else:
        model_path = str(project_root / "intervention" / "models" / args.model_name / "final_model")
    
    # Verify path exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    model_map = {
        'gpt2': 'gpt2',
        'gemma': 'google/gemma-2b',
        'deepseek': 'deepseek-ai/deepseek-coder-1.3b-base'
    }
    
    if args.verbose:
        print(f"Loading model and tokenizer from {model_path}")
    
    if args.model_name == 'gemma':
        # Loading gemma2 config requires special authorization
        login("xxxxxxx")
    
    hf_model_name = model_map.get(args.model_name, args.model_name)
    
    # Load tokenizer from model path with local_files_only=True
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        local_files_only=True,
        trust_remote_code=False
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config from HuggingFace for model structure
    config = AutoConfig.from_pretrained(
        hf_model_name, 
        num_labels=num_labels, 
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Load model from same path with local_files_only=True
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        config=config, 
        ignore_mismatched_sizes=True,
        local_files_only=True,
        trust_remote_code=False
    )
    
    model_wrapper = LanguageModel(model, tokenizer=tokenizer, device_map=device)
    model_wrapper.eval()
    
    # Get layer information
    num_layers = model.config.num_hidden_layers
    
    return model_wrapper, tokenizer, num_layers