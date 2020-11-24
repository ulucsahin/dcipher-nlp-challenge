import train
from json_manager import JSONManager

if __name__ == '__main__':
    json_path = "data/wos2class.json"
    json_mng = JSONManager(json_path)
    json_mng.split_test_train(0.8)
    print(len(json_mng.data_test))


