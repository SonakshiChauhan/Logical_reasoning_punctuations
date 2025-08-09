import sys
import os
from pathlib import Path
import torch
import pickle
import numpy as np


def get_project_root():
    """Get the project root directory."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    print(f"Using project root: {PROJECT_ROOT}")
    return PROJECT_ROOT


def set_device(device_arg):
    """Set and return the appropriate device."""
    device = torch.device(device_arg)
    return device