from huggingface_hub import HfApi, login, create_repo
from pathlib import Path


def save_to_huggingface(output_dir, args):
    """Upload results to HuggingFace Hub."""
    # Login using token
    hardcoded_token = "xxxxxxxx"  # Replace this!
    
    login(token=hardcoded_token)
    
    api = HfApi()
    
    # Create repository name
    repo_id = f"{api.whoami()['name']}/intervention-results-{args.model_name}-{args.intervention_type}"
    
    print(f"Creating/using repository: {repo_id}")
    
    # Create repo if it doesn't exist
    create_repo(
        repo_id=repo_id, 
        repo_type="dataset", 
        exist_ok=True,
        private=False  # Set to True if you want private repo
    )
    
    # Upload all files in output directory
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            print(f"Uploading {file_path.name}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=repo_id,
                repo_type="dataset"
            )
    
    print(f"✅ Results uploaded successfully!")
    print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")