import torch
import pandas as pd
import numpy as np
import json

def train():
    print(torch.__version__)

    # open the file
    f = open("data/wos2class.json", )

    data = json.load(f)
    print(data[0])
    #json_data = json.loads()