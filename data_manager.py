import re
import json
import nltk
import random


class DataManager(object):
    """
    This class is responsible from json operations
    """

    def __init__(self, config, augmenter):
        self.config = config
        self.augmenter = augmenter
        self.data_test = []
        self.data_train = []
        self.word_frequencies = {}
        self.vocab_size = -1
        self.data = None
        self.read_json(self.config.json_path)

    def preprocess_data(self):
        self.shuffle()
        self.apply_regex()
        # self.remove_bad_data() # disabled
        self.calculate_word_frequencies()
        self.calculate_vocabulary_size()
        self.tokenize_sentences()
        self.split_test_train(self.config.train_ratio, self.config.split_method)
        self.pad_sentences(self.config.title_pad_length, self.config.abstract_pad_length)
        if self.config.augment:
            self.augment_train_data()

    def read_json(self, path):
        with open(path) as json_file:
            self.data = json.load(json_file)

    def shuffle(self):
        # in place
        random.shuffle(self.data)

    # def read_test_train_data(self, train_path, test_path):
    #     with open(train_path) as json_file:
    #         self.data_train = json.load(json_file)
    #
    #     with open(test_path) as json_file:
    #         self.data_test = json.load(json_file)

    def apply_regex(self):
        """
        Insert space before special characters such as "," or "."  etc.
        Convert to lowercase
        Remove stopwords (English)
        """

        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')

        for i in range(len(self.data)):
            # get title and abstract
            current_title = self.get_title_from_idx(i)
            current_abstract = self.get_abstract_from_idx(i)

            # apply regex
            # pattern = re.compile(r"([.()!\-\/@\[\]\%])")
            pattern = re.compile(r"([.,])")
            stopword_pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
            # current_title = pattern.sub(" \\1 ", current_title).lower()
            # current_abstract = pattern.sub(" \\1 ", current_abstract).lower()
            current_title = pattern.sub("", current_title).lower()
            current_abstract = pattern.sub("", current_abstract).lower()

            # remove stopwords using regex
            current_title = stopword_pattern.sub('', current_title)
            current_abstract = stopword_pattern.sub('', current_abstract)

            # assign
            self.data[i]["Title"] = current_title
            self.data[i]["Abstract"] = current_abstract

    def augment_train_data(self):
        """
        Augmenter is slow. This would be a problem with big data.
        """
        # do not augment on evaluation dataset
        original_len = len(self.data_train)
        for i in range(len(self.data_train)):
            if i % 100 == 0:
                print(f"Augmenting train data, progress: {i} / {original_len}")
            title = self.data_train[i]["Title"]
            abstract = self.data_train[i]["Abstract"]
            label = self.data_train[i]["Label"]

            title = self.augmenter.augment(title)
            abstract = self.augmenter.augment(abstract)

            self.data_train.append({"Title": title, "Abstract": abstract, "Label": label})
        print(f"Train data amount after augmenting: {len(self.data_train)}")

    def split_test_train(self, train_ratio, split_method):
        """
        :param train_ratio: train data / all data ratio
        :param split_method: 0: split randomly, 1: split each label individually to ensure numbers given in github
        :return: None
        """
        if split_method == 0:
            split_index = int(len(self.data) * train_ratio)
            self.data_train = self.data[0:split_index]
            self.data_test = self.data[split_index:len(self.data)]
            assert(len(self.data_train) + len(self.data_test) == len(self.data))
        elif split_method == 1:
            label_split_data = []
            for i in range (len(self.config.label_dict)):
                label_split_data.append(0)
                label_split_data[i] = []

            # split into labels
            for data in self.data:
                label_idx = self.config.label_dict[data["Label"]]
                label_split_data[label_idx].append(data)

            # split into train-test
            for data in label_split_data:
                split_index = int(len(data) * train_ratio)
                [self.data_train.append(row) for row in data[0:split_index]]
                [self.data_test.append(row) for row in data[split_index:len(data)]]

            # shuffle
            random.shuffle((self.data_train))
            random.shuffle((self.data_test))

    def create_train_test_jsonfile(self):
        with open(self.config.train_data_path, 'w') as outfile:
            json.dump(self.data_train, outfile)
        with open(self.config.test_data_path, 'w') as outfile:
            json.dump(self.data_test, outfile)

    def remove_bad_data(self):
        """
        If there are (and there are) some data instances with empty title or abstract fields, remove them.
        """
        min_title_len = 9e10
        min_abstract_len = 9e10
        indexes_to_remove = []
        for i in range(len(self.data)):
            title_len = len(self.data[i]["Title"])
            if title_len < min_title_len:
                min_title_len = title_len

            abstract_len = len(self.data[i]["Abstract"])
            if abstract_len < min_abstract_len:
                min_abstract_len = abstract_len

            if title_len == 0 or abstract_len == 0:
                if i not in indexes_to_remove:
                    indexes_to_remove.append(i)

        # start deleting from the end of list to avoid problems with shifting indexes
        indexes_to_remove.reverse()
        for index in indexes_to_remove:
            print("Deleting index: ", index)
            del self.data[index]

        print("Min Title len:", min_title_len)
        print("Min Abstract len:", min_abstract_len)

    def tokenize_sentences(self):
        for i in range(len(self.data)):
            self.data[i]["Title"] = self.get_title_from_idx(i).split(" ")
            self.data[i]["Abstract"] = self.get_abstract_from_idx(i).split(" ")

    def get_title_from_idx(self, idx):
        return self.data[idx]["Title"]

    def get_abstract_from_idx(self, idx):
        return self.data[idx]["Abstract"]

    def pad_sentences(self, desired_length_title, desired_length_abstract):
        # data is assumed to be tokenized at this point
        pad_word = ["<pad>"]
        for i in range(len(self.data)):
            title = self.get_title_from_idx(i)
            abstract = self.get_abstract_from_idx(i)

            # pad title
            if len(title) > desired_length_title:
                title = title[0:desired_length_title]
            elif len(title) < desired_length_title:
                title += pad_word * (desired_length_title - len(title))

            # pad abstract
            if len(abstract) > desired_length_abstract:
                abstract = abstract[0:desired_length_abstract]
            elif len(abstract) < desired_length_abstract:
                abstract += pad_word * (desired_length_abstract - len(abstract))

            self.data[i]["Title"] = title
            self.data[i]["Abstract"] = abstract

    def calculate_word_frequencies(self):
        self.word_frequencies = {}
        for i in range(len(self.data)):
            title = self.get_title_from_idx(i).split(' ')
            abstract = self.get_abstract_from_idx(i).split(' ')

            for word in title:
                try:
                    self.word_frequencies[word] += 1
                except KeyError:
                    self.word_frequencies[word] = 1

            for word in abstract:
                try:
                    self.word_frequencies[word] += 1
                except KeyError:
                    self.word_frequencies[word] = 1

    def calculate_vocabulary_size(self):
        self.vocab_size = len(self.word_frequencies)

    def count_labels(self):
        counts_train = [0, 0]
        counts_test = [0, 0]
        for i in range(len(self.data_train)):
            if self.data_train[i]["Label"] == "Material Science":
                counts_train[0] += 1
            elif self.data_train[i]["Label"] == "Chemistry":
                counts_train[1] += 1

        for i in range(len(self.data_test)):
            if self.data_test[i]["Label"] == "Material Science":
                counts_test[0] += 1
            elif self.data_test[i]["Label"] == "Chemistry":
                counts_test[1] += 1

        print(f"Train Num Material: {counts_train[0]}")
        print(f"Train Num Chemistry: {counts_train[1]}")

        print(f"Test Num Material: {counts_test[0]}")
        print(f"Test Num Chemistry: {counts_test[1]}")
