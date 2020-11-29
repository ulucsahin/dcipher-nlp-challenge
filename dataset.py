from torch.utils.data import Dataset
import json
import numpy as np


class WOSDataset(Dataset):
    def __init__(self, config, embedder, is_train):
        self.config = config
        self.embedder = embedder
        self.is_train = is_train
        self.data = None

        if is_train:
            self.data_path = config.train_data_path
        else:
            self.data_path = config.test_data_path

        self.read_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = self.get_title_from_idx(idx)
        abstract = self.get_abstract_from_idx(idx)
        label = self.get_label_from_idx(idx)

        title_embeddings = self.embedder.get_word_embeddings([title]) # (1, token amount, embedding size)
        abstract_embeddings = self.embedder.get_word_embeddings([abstract]) # (1, token amount, embedding size)
        label = self.config.label_dict[label]

        return title_embeddings, abstract_embeddings, label

    def indices_to_one_hot(self, data, nb_classes):
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

    def read_data(self):
        with open(self.data_path) as json_file:
            self.data = json.load(json_file)

    def get_title_from_idx(self, idx):
        return self.data[idx]["Title"]

    def get_abstract_from_idx(self, idx):
        return self.data[idx]["Abstract"]

    def get_label_from_idx(self, idx):
        return self.data[idx]["Label"]

    def get_vocab_size(self):
        return


