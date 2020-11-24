import re
import json

class JSONManager(object):
    """
    This class is responsible about json operations
    """

    # for readability
    json_path = None
    data = None
    data_train = None
    data_test = None

    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self.read_json(json_path)
        #self.preprocess_text_data()
        #self.frequencies = self.get_word_frequencies()
        #self.word_to_idx = self.get_word_to_idx_mapping()
        #self.idx_to_word = self.get_idx_to_word_mapping()
        #self.vocab_size = self.get_vocab_size()

    def read_json(self, path):
        with open(path) as json_file:
            data = json.load(json_file)

        return data

    def preprocess_text_data(self):
        """
        Insert space before special characters such as "," or "."  etc.
        TODO: make everything lowercase
        """
        # for i in range(len(self.data)):
        #     for j in range(len(self.data[i]["descriptions"])):
        #         desc = self.get_description_from_idx_with_idx(i, j)
        #         desc = re.sub(r"([^a-zA-Z])", r" \1 ", desc)
        #         desc = re.sub('\s{2,}', ' ', desc)
        #         self.data[i]["descriptions"][j]["text"] = desc
        pass

    def split_test_train(self, train_ratio):
        split_index = int(len(self.data) * train_ratio)
        self.data_train = self.data[0:split_index]
        self.data_test = self.data[split_index:len(self.data)]

        assert(len(self.data_train) + len(self.data_test) == len(self.data))