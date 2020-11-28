import numpy as np
import torch


def random_seed(seed_value):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed_value)


def load_model(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    return model

def load_accuracies(model_path):
    checkpoint = torch.load(model_path)
    train_accs = checkpoint["train_accs"]
    test_accs = checkpoint["test_accs"]

    return train_accs, test_accs
