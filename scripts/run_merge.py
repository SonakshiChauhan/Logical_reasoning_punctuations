import argparse
from src.utils.merge_lora import merge_lora_adapter

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with Hugging Face model")

    parser.add_argument('--zip_path', type=str, required=True, help='Path to zipped LoRA adapter')
    parser.add_argument('--hf_token', type=str, required=True, help='Hugging Face access token')
    parser.add_argument('--output_path', type=str, required=True, help='Where to save the merged model')
    parser.add_argument('--model_name', type=str, default='deepseek', help='Model name (deepseek, gemma, gpt2)')
    parser.add_argument('--num_labels', type=int, default=3, help='Number of classification labels')
    parser.add_argument('--hf_model_id', type=str, help='Hugging Face model ID (optional, auto-set based on model_name)')
    
    args = parser.parse_args()

    # Automatically set hf_model_id based on model_name if not provided
    if args.hf_model_id is None:
        if args.model_name == 'gemma':
            args.hf_model_id = 'google/gemma-2b'
        elif args.model_name == 'deepseek':
            args.hf_model_id = 'deepseek-ai/deepseek-coder-1.3b-base'
        else:
            raise ValueError(f"Unknown model_name: {args.model_name}. Supported: deepseek, gemma, gpt2")

    print(f"Using model: {args.model_name}")
    print(f"HuggingFace model ID: {args.hf_model_id}")

    merge_lora_adapter(
        zip_path=args.zip_path,
        hf_model_id=args.hf_model_id,
        hf_token=args.hf_token,
        output_path=args.output_path,
        num_labels=args.num_labels
    )


if __name__ == '__main__':
    main()
