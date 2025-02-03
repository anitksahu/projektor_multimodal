# otdd/pytorch/utils.py

import torch

def preprocess_input(x):
    """
    Generalized preprocessing function.

    If x is a 3-dimensional tensor (e.g., a single image), unsqueeze to add a batch dimension.
    Otherwise, returns x unchanged.
    """
    if isinstance(x, torch.Tensor) and x.dim() == 3:
        return x.unsqueeze(0)
    return x

