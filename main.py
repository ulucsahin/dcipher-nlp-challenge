from embedding_manager import Embedder
from dataset import WOSDataset
from CNN import CNN
from train import Trainer
from config import Config
import random
import test
from data_manager import DataManager
import nlpaug.augmenter.word as nas

from util import *

if __name__ == '__main__':
    random_seed(random.randint(0, 100000))

    # create configuration file
    config = Config()

    # create text embedder
    embedder = Embedder(config.embedding_length)

    augmenter = None
    if config.augment:
        augmenter = nas.AntonymAug()

    # preprocess data and create wos2class.text.json and wos2class.train.json
    if not config.use_existing_data:
        data_manager = DataManager(config, augmenter)
        data_manager.preprocess_data()
        data_manager.create_train_test_jsonfile()
        data_manager.count_labels()

    # create dataset and dataloaders
    train_dataset = WOSDataset(config, embedder, is_train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = iter(train_dataloader)

    test_dataset = WOSDataset(config, embedder, is_train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    test_iter = iter(test_dataloader)

    # create model
    model = CNN(config)

    if config.train:
        trainer = Trainer(config, train_dataset, test_dataset)
        trainer.current_lr = config.lr

        # restore model if continuing training
        if config.continue_train:
            # load from checkpoint
            checkpoint = torch.load(config.model_to_load_path)
            model.load_state_dict(checkpoint['model'])
            trainer.train_accs = checkpoint['train_accs']
            trainer.test_accs = checkpoint['test_accs']
            trainer.current_epoch = checkpoint['epoch']
            trainer.best_test_acc = checkpoint['best_test_acc']
            trainer.current_lr = checkpoint['current_lr']

        # train
        trainer.begin_training(model),

        # evaluate after training ends
        checkpoint = torch.load(config.model_to_load_path)
        model.load_state_dict(checkpoint['model'])
        test.begin_evaluation(config, model, test_dataset)
    # evaluate only
    else:
        # restore model to evaluate
        checkpoint = torch.load(config.model_to_load_path)
        model.load_state_dict(checkpoint['model'])

        # evaluate
        test.begin_evaluation(config, model, test_dataset)
