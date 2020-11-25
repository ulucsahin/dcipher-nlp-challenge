import re
import json

class JSONManager(object):
    """
    This class is responsible from json operations
    """

    # for readability
    json_path = None
    data = None
    data_train = None
    data_test = None

    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self.read_json(json_path)
        self.remove_bad_data()

    def read_json(self, path):
        with open(path) as json_file:
            data = json.load(json_file)

        return data

    def split_test_train(self, train_ratio):
        split_index = int(len(self.data) * train_ratio)
        self.data_train = self.data[0:split_index]
        self.data_test = self.data[split_index:len(self.data)]
        assert(len(self.data_train) + len(self.data_test) == len(self.data))

    def create_train_test_jsonfile(self, train_ratio):
        self.split_test_train(train_ratio)
        with open('data/wos2class.train.json', 'w') as outfile:
            json.dump(self.data_train, outfile)
        with open('data/wos2class.test.json', 'w') as outfile:
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
            if(title_len < min_title_len):
                min_title_len = title_len

            abstract_len = len(self.data[i]["Abstract"])
            if(abstract_len < min_abstract_len):
                min_abstract_len = abstract_len

            if(title_len == 0 or abstract_len == 0):
                if(i not in indexes_to_remove):
                    indexes_to_remove.append(i)

        # start deleting from the end of list to avoid problems with shifting indexes
        indexes_to_remove.reverse()
        for index in indexes_to_remove:
            print("Deleting index: ", index)
            del self.data[index]




        print("Min Title len:", min_title_len)
        print("Min Abstract len:", min_abstract_len)

