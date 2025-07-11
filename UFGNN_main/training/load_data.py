import pandas as pd
import csv
import numpy as np
import os
from torch import nn
import torch
import random
import pickle
def load_data():

    code_num = 171
    fts = 5
    f = open(r'../data/data4.csv')
    df = pd.read_csv(f, header=None)
    data = df.iloc[:, 0:-1].values
    eod_data = data.reshape(-1, code_num, fts)
    data_label = df.iloc[:, -1].values
    ground_truth = data_label.reshape(code_num, -1)

    return eod_data, ground_truth


def load_dataset2(DEVICE, rnn_length):
    with open(r'../SP500/x_numerical.pkl', 'rb') as handle:
        markets = pickle.load(handle)
    with open(r'../SP500/y_.pkl', 'rb') as handle:
        y_load = pickle.load(handle)


    markets = markets.astype(np.float64)
    x = torch.tensor(markets, device=DEVICE)
    x.to(torch.double)

    y = torch.tensor(y_load, device=DEVICE)
    y = (y>0).to(torch.long)

    x_train = x[: -140]
    x_eval = x[-140 - rnn_length: -70]
    x_test = x[-70 - rnn_length:]

    y_train = y[: -140]
    y_eval = y[-140 - rnn_length: -70]
    y_test = y[-70 - rnn_length:]


    return x_train,x_eval,x_test,y_train,y_eval,y_test

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

