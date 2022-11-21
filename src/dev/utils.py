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

def collapse(sequence: Sequence, blank_symbol: int = 0):
    curr_idx = None
    output = []
    for idx in sequence:
        if idx == blank_symbol:
            curr_idx = None
        elif idx != curr_idx:
            output.append(idx)
            curr_idx = idx
    return output

def greedy_decode(log_probs: torch.Tensor, decode_dict: Dict[int, str]):
    preds_id_full = torch.argmax(log_probs, dim = 2)
    preds_id_full = preds_id_full.detach().cpu().numpy()
    
    preds_ids = []
    preds_full = []
    preds = []
    
    for pred_id in preds_id_full:
        pred_collapsed = collapse(pred_id)
        preds_ids.append(pred_collapsed)
        pred_sequence_full = "".join([decode_dict[idx] for idx in pred_id])
        preds_full.append(pred_sequence_full)
        pred_sequence = "".join([decode_dict[idx] for idx in pred_collapsed])
        preds.append(pred_sequence)
    return preds_ids, preds_full, preds

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    probs = torch.rand(5, 75, 10)
    char_dict = {0:"*"}
    for i in range (1,10):
        char_dict[i] = chr(96 + i)
    log_probs = torch.softmax(probs, dim = 2)
    print(greedy_decode(probs, char_dict))
    