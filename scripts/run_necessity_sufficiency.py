import argparse
from src.ablations.necessity_sufficiency import run_necessity_sufficiency
from src.utils.helpers import get_project_root, set_device
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = set_device(args.device)
    project_root = get_project_root()
    run_necessity_sufficiency(args, project_root, device)