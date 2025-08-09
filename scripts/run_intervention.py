import sys
import os
import argparse
import torch
from pathlib import Path

# Add the project root to Python path so we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from your src/ modules
from src.utils.helpers import get_project_root, set_device
from src.utils.visualisation import save_layer_accuracies
from src.interventions.token_intervention import run_intervention


def main():
    """Main function to run intervention experiments."""
    parser = argparse.ArgumentParser(description="Run multi-layer causal interventions on language models.")
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size for processing.")
    parser.add_argument("--model_path", type=str, default=None, help="Custom path to model directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda, cpu, mps).")
    parser.add_argument("--intervention_type", type=str, required=True, help="Type of intervention (e.g., adjective, subject, full_sentence, two_sentence).")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (e.g., GPT2, Gemma, DeepSeek).")
    parser.add_argument("--dataset_dir", type=str, default="/Datasets_intervention", help="Directory containing intervention datasets.")
    parser.add_argument("--output_dir", type=str, default="intervention_results", help="Directory to save output CSV/PKL files.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging and progress updates.")
    parser.add_argument("--output_logits", action='store_true', help="Save logits of final predictions.")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers in the model.")
    
    args = parser.parse_args()

    if args.verbose:
        print(f"Running intervention with args: {vars(args)}")

    # Setup device and project root using utility functions
    device = set_device(args.device)
    project_root = get_project_root()
    
    if args.device == "mps" and args.verbose:
        print("Using Apple Silicon GPU (MPS) for model inference.")
    elif args.device == "cuda" and args.verbose:
        print("Using NVIDIA GPU for model inference.")
    elif args.device == "cpu" and args.verbose:
        print("Using CPU for model inference.")

    # Run the intervention using the extracted function
    layer_accuracies = run_intervention(args, project_root, device)

    if args.verbose:
        print("Intervention run finished.")
        print(f"Layer accuracies: {layer_accuracies}")

    # Save visualization using extracted function
    save_layer_accuracies(layer_accuracies, args, project_root)

    print("Intervention complete!")


if __name__ == "__main__":
    main()