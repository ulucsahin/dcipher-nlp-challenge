import train
from json_manager import JSONManager
from embedding_manager import Embedder
from data_manager import WOSDataset
from uluc_model1 import CNN
import torch
import train


import re

if __name__ == '__main__':
    # args
    json_path = "data/wos2class.json"
    train_data_path = "data/wos2class.train.json"
    test_data_path = "data/wos2class.test.json"
    train_ratio = 0.8
    batch_size = 16
    learning_rate = 2e-5
    output_size = 2
    in_channels = 1
    out_channels = 100
    kernel_heights = [1]
    stride = 1
    padding = 0
    keep_probab = 0.75
    embedding_length = 300

    # create json manager and read data
    json_mng = JSONManager(json_path)
    json_mng.create_train_test_jsonfile(train_ratio)
    #json_mng.split_test_train(train_ratio)

    # create embedder
    embedder = Embedder()

    # create dataset and dataloaders
    train_dataset = WOSDataset(train_data_path, embedder)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # train_iter = iter(train_dataloader)

    test_dataset = WOSDataset(test_data_path, embedder)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_iter = iter(test_dataloader)


    # create model
    vocab_size = train_dataset.vocab_size
    model = CNN(batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length)
    num_epochs = 5
    for epoch in range(num_epochs):
        train.train_model(model, train_dataset, batch_size, epoch)

        print("END OF EPOCH ", epoch)








