import pandas as pd
from pathlib import Path


def get_dataset(args):
    """Load the appropriate dataset based on intervention type."""
    dataset_folder = Path(args.dataset_dir)  # Use command line argument
    
    dataset_mapping = {
        "two_sentence_on_dot": 'two_sent_check.csv',
        "two_sentence_on_sentence": 'two_sent_check.csv',
        "first_sentence_on_dot": 'rule_taker_full_sentence.csv',
        "first_sentence_on_sentence": 'rule_taker_full_sentence.csv',
        "subject": 'rule_taker_subject.csv',
        "adjective": 'rule_taker_adjective.csv'
    }
    
    if args.intervention_type not in dataset_mapping:
        raise ValueError(f"Unknown intervention type: {args.intervention_type}")
    
    dataset_path = dataset_folder / dataset_mapping[args.intervention_type]

    # Load the dataset
    if args.verbose:
        print(f"Loading dataset from {dataset_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    
    df = pd.read_csv(dataset_path, header=0)
    print(f"Loaded unfiltered dataset with {len(df)} rows.")
    return df
