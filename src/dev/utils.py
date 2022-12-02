import torch
from typing import Dict, Sequence
import os
import torch
import numpy as np
import random

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("Model Successfully saved at", path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    