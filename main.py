from json_manager import JSONManager
from embedding_manager import Embedder
from data_manager import WOSDataset
from uluc_model1 import CNN
import torch
import train
from config import Config
import nlpaug.augmenter.word as nas
import numpy as np
import random
import test

def random_seed(seed_value):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed_value)

def load_model(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    return model

if __name__ == '__main__':
    random_seed(random.randint(0, 100000))

    # create configuration file
    config = Config()

    # data augmenter
    augmenter = nas.AntonymAug()

    # create json manager and read data
    json_mng = JSONManager(config, augmenter)
    json_mng.count_labels()

    # create embedder
    embedder = Embedder(config.embedding_length)

    # create dataset and dataloaders
    train_dataset = WOSDataset(config.train_data_path, embedder, True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = iter(train_dataloader)

    test_dataset = WOSDataset(config.test_data_path, embedder, False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    test_iter = iter(test_dataloader)

    # create model
    model = CNN(config)

    if config.continue_train:
        model = load_model(model, config.model_to_load_path)

    # train
    if (config.train):
        # optimizer
        optim = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        train = train.Trainer(config, train_dataset, test_dataset)
        for epoch in range(config.epochs):
            train.train_model(model, optim, epoch)
    # evaluate
    else:
        results = test.evaluate_model(model, test_dataset, config.batch_size)
        print(f"Accuracy on test dataset: {results['accuracy']}")
        print(results["confusion_matrix"])
        print(results["performance_measures"]["false_negatives"])
        print(results["performance_measures"]["precision"][0])













