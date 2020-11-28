import os


class Config(object):
    # paths
    json_path = os.path.join('data', 'wos2class.json')
    train_data_path = os.path.join('data', 'wos2class.train.json')
    test_data_path = os.path.join('data', 'wos2class.test.json')
    model_to_load_path = os.path.join('model', 'BestModel.pth')  # path to model which will be used in evaluation or to continue training

    # labels
    label_dict = {"Material Science": 0, "Chemistry": 1}

    # optimizer parameters
    lr = 0.0005
    weight_decay = 1e-5

    # model hyperparameters
    train_ratio = 0.8
    batch_size = 32
    output_size = 2
    in_channels = 1
    out_channels = 25
    kernel_heights = [3, 4, 5]
    stride = 1
    padding = 0
    keep_probab = 0.6  # dropout keep probability
    embedding_length = 300
    epochs = 200
    print_frequency = 25

    # data settings
    title_pad_length = 50 # number of tokens in each title after padding
    abstract_pad_length = 300  # number of tokens in each abstract after padding
    split_method = 1  # 0: randomly splitting data, 1: splitting label-wise
    num_classes = 2

    # run settings
    train = True  # train or test
    continue_train = False  # whether continue from saved model or start from beginning
    use_existing_data = False  # generate train-test data from wos2class.json or use pre-computed train-test data
    augment = False  # augment train data or not, only applicable if use_existing_data is False
