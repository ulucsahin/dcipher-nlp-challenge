import numpy as np
import torch


def random_seed(seed_value):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed_value)

def load_accuracies(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    train_accs = checkpoint['train_accs']
    test_accs = checkpoint['test_accs']

    return train_accs, test_accs