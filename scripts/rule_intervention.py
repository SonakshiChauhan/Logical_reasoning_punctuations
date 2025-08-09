import sys
import os
import argparse
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import get_project_root, set_device
from src.utils.visualisation import save_layer_accuracies
from src.interventions.rule_intervention import run_rule_intervention


def main():
    """Main function to run rule intervention experiments."""
    parser = argparse.ArgumentParser(description="Run rule-based interventions on language models.")
    
    # Standard arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
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
    
    # Rule-specific arguments
    parser.add_argument("--rule_type", type=str, required=True, 
                       choices=['all_rule', 'if_then_rule'],
                       help="Type of rule intervention.")
    parser.add_argument("--max_samples", type=int, default=200, 
                       help="Maximum number of samples to process (0 for all).")
    
    args = parser.parse_args()

    if args.verbose:
        print(f"Running rule intervention with args: {vars(args)}")

    # Setup
    device = set_device(args.device)
    project_root = get_project_root()
    
    if args.verbose:
        print(f"Using device: {device}")
        print(f"Rule type: {args.rule_type}")
        print(f"Model: {args.model_name}")

    # Run rule intervention
    layer_accuracies = run_rule_intervention(args, project_root, device)

    if args.verbose:
        print("Rule intervention completed!")
        print(f"Final layer accuracies: {layer_accuracies}")

    # Create visualization
    save_layer_accuracies(layer_accuracies, args, project_root)
    
    print("Rule intervention experiment complete!")
    
    if __name__ == "__main__":
        main()