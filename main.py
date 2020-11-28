from embedding_manager import Embedder
from dataset import WOSDataset
from CNN import CNN
import train
from config import Config
import random
import test
import nlpaug.augmenter.word as nas
from util import *

if __name__ == '__main__':
    random_seed(random.randint(0, 100000))

    # create configuration file
    config = Config()

    # create text embedder
    embedder = Embedder(config.embedding_length)

    # create dataset and dataloaders
    train_dataset = WOSDataset(config, embedder, is_train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = iter(train_dataloader)

    test_dataset = WOSDataset(config, embedder, is_train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    test_iter = iter(test_dataloader)

    # create model
    model = CNN(config)

    if config.continue_train:
        model = load_model(model, config.model_to_load_path)

    # train
    if config.train:
        trainer = train.Trainer(config, train_dataset, test_dataset)

        if config.augment:
            augmenter = nas.AntonymAug()
            trainer.assign_augmenter(augmenter)

        # optimizer
        optim = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        for epoch in range(config.epochs):
            trainer.train_model(model, optim, epoch)
    # evaluate
    else:
        model = load_model(model, config.model_to_load_path)
        train_accs, test_accs = load_accuracies(config.model_to_load_path)

        results = test.evaluate_model(model, test_dataset, config.batch_size)
        print(f"Accuracy on test dataset: {results['accuracy']}")
        print(results["confusion_matrix"])

        test.plot_accuracies(train_accs, test_accs)










