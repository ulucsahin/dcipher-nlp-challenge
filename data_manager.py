import torch
from torch.utils.data import Dataset
import json
import re

class WOSDataset(Dataset):
    # for readability
    data_path = ""
    data = None
    word_frequencies = {}
    vocab_size = -1
    embedder = None
    label_mapping = {"Material Science": 0, "Chemistry": 1}

    def __init__(self, data_path, embedder):
        self.data_path = data_path
        self.embedder = embedder
        self.read_data()
        self.preprocess_text_data()
        # self.tokenize_data()
        # self.calculate_word_frequencies()
        self.calculate_vocabulary_size()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        title = self.get_title_from_idx(idx)
        abstract = self.get_abstract_from_idx(idx)
        label = self.get_label_from_idx(idx)

        title_embeddings = self.embedder.get_embeddings([title])
        abstract_embeddings = self.embedder.get_embeddings([abstract])

        return title_embeddings, abstract_embeddings, self.label_mapping[label]

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

    def tokenize_data(self):
        for i in range(len(self.data)):
            self.data[i]["Title"] = self.get_title_from_idx(i).split(" ")
            self.data[i]["Abstract"] = self.get_abstract_from_idx(i).split(" ")

    def get_title_from_idx(self, idx):
        return self.data[idx]["Title"]

    def get_abstract_from_idx(self, idx):
        return self.data[idx]["Abstract"]

    def get_label_from_idx(self, idx):
        return self.data[idx]["Label"]

    def get_vocab_size(self):
        return

    def calculate_word_frequencies(self):
        self.word_frequencies = {}
        for i in range(len(self.data)):
            title = self.get_title_from_idx(i)
            abstract = self.get_abstract_from_idx(i)

            for word in title:
                try:
                    self.word_frequencies[word] += 1
                except:
                    self.word_frequencies[word] = 1

            for word in abstract:
                try:
                    self.word_frequencies[word] += 1
                except:
                    self.word_frequencies[word] = 1

    def calculate_vocabulary_size(self):
        word_frequencies = {}
        for i in range(len(self.data)):
            title = self.get_title_from_idx(i).split(" ")
            abstract = self.get_abstract_from_idx(i).split(" ")

            for word in title:
                try:
                    word_frequencies[word] += 1
                except:
                    word_frequencies[word] = 1

            for word in abstract:
                try:
                    word_frequencies[word] += 1
                except:
                    word_frequencies[word] = 1

        self.vocab_size = len(word_frequencies)
