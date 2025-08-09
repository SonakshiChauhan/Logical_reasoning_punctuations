import os
import zipfile
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import PeftModel
from huggingface_hub import login


def unzip_adapter(zip_path, extract_to):
    """Unzips the adapter zip file to a target location."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to


def merge_lora_adapter(zip_path, hf_model_id, hf_token, output_path, num_labels=3):
    # Login to Hugging Face
    login(token=hf_token, add_to_git_credential=False)

    # Unzip adapter
    adapter_path = unzip_adapter(zip_path, extract_to="./lora_adapters/tmp_adapter")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load config
    config = AutoConfig.from_pretrained(
        hf_model_id,
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id
    )

    # Load base model
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    # Merge LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
        local_files_only=True
    )
    merged_model = model.merge_and_unload()

    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to: {output_path}")

    # Cleanup
    del base_model, model, merged_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
