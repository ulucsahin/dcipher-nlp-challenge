import numpy as np
import torch


def random_seed(seed_value):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed_value)