import torch
from torch.utils.data import Dataset
import json
import re
import numpy as np

num_classes = 2

class WOSDataset(Dataset):
    label_mapping = {"Material Science": 1, "Chemistry": 0}

    def __init__(self, data_path, embedder, is_train):
        self.data_path = data_path
        self.embedder = embedder
        self.read_data()
        self.is_train = is_train


    def __len__(self):
        return len(self.data)

    # word embedding version
    def __getitem__(self, idx):
        title = self.get_title_from_idx(idx)
        abstract = self.get_abstract_from_idx(idx)
        label = self.get_label_from_idx(idx)

        title_embeddings = self.embedder.get_word_embeddings([title]) # (1, token amount, embedding size)
        abstract_embeddings = self.embedder.get_word_embeddings([abstract]) # (1, token amount, embedding size)
        label = self.label_mapping[label]

        # convert to onehot
        onehot = False
        if onehot:
            label = self.indices_to_one_hot(label, num_classes)


        return title_embeddings, abstract_embeddings, label

    def indices_to_one_hot(self, data, nb_classes):
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

    def read_data(self):
        with open(self.data_path) as json_file:
            self.data = json.load(json_file)


    def preprocess_text_data(self):
        """
        Insert space before special characters such as "," or "."  etc.
        TODO: make everything lowercase
        """

        for i in range(len(self.data)):
            # get title and abstract
            current_title = self.get_title_from_idx(i)
            current_abstract = self.get_abstract_from_idx(i)

            # apply regex
            pattern = re.compile(r"([.()!\-\/@\[\]\%])")
            current_title = pattern.sub(" \\1 ", current_title)
            current_abstract = pattern.sub(" \\1 ", current_abstract)

            # assign
            self.data[i]["Title"] = current_title
            self.data[i]["Abstract"] = current_abstract

    def get_title_from_idx(self, idx):
        return self.data[idx]["Title"]

    def get_abstract_from_idx(self, idx):
        return self.data[idx]["Abstract"]

    def get_label_from_idx(self, idx):
        return self.data[idx]["Label"]

    def get_vocab_size(self):
        return


