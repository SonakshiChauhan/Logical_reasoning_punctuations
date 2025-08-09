import sys
import os
import argparse
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import get_project_root, set_device
from src.utils.visualisation import create_symmetric_heatmap
from src.ablations.layer_swap import run_layer_swap, process_layer_swap


def main():
    """Main function to run rule intervention experiments."""
    parser = argparse.ArgumentParser(description="Run rule-based interventions on language models.")
    
    # Standard arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--swap_type", type=int, default=1, help="Rule type for checking layer swap")
    parser.add_argument("--model_path", type=str, default=None, help="Custom path to model directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use (cuda, cpu, mps).")
    parser.add_argument("--model_name", type=str, required=True, choices=['gpt2', 'gemma', 'deepseek'],
                       help="Name of the model.")
    parser.add_argument("--dataset_dir", type=str, default="Datasets_intervention", 
                       help="Directory containing rule datasets.")
    parser.add_argument("--output_dir", type=str, default="rule_intervention_results", 
                       help="Directory to save output files.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging.")

    
    args = parser.parse_args()

    if args.verbose:
        print(f"Running swap with args: {vars(args)}")

    # Setup
    device = set_device(args.device)
    project_root = get_project_root()
    
    if args.verbose:
        print(f"Using device: {device}")
        print(f"Rule type: {args.rule_type}")
        print(f"Model: {args.model_name}")

    mean_layer = run_layer_swap(args, project_root, device)

    if args.verbose:
        print("Swap completed!")

    create_symmetric_heatmap(args,mean_layer, project_root)
    
    print("Rule intervention experiment complete!")
    
    if __name__ == "__main__":
        main()