# This is based on the example presented by Shibo Li in class

import torch
import pandas as pd
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

#from misc import *

import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm, trange

device = torch.device('cuda:0')

torch.manual_seed(222)
np.random.seed(333)

output_col = "genuine"
notes_train = pd.read_csv("bank-note/train.csv", names=['variance', 'skew', 'cutosis', 'entropy', output_col])
notes_test = pd.read_csv("bank-note/test.csv", names=['variance', 'skew', 'cutosis', 'entropy', output_col])

notes_train.convert_dtypes()
notes_test.convert_dtypes()

# replace 0 with -1 in the output
notes_train[output_col] = notes_train[output_col].replace(0, -1)
notes_test[output_col] = notes_test[output_col].replace(0, -1)

trainY = notes_train[output_col].to_numpy()
trainX = notes_train.drop(output_col, axis=1).to_numpy()

testY = notes_test[output_col].to_numpy()
testX = notes_test.drop(output_col, axis=1).to_numpy()

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        super(SimpleDataset, self).__init__()
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index,:], self.y[index,:]

    def __len__(self,):
        return self.X.shape[0]
        
dataset_train = SimpleDataset(trainX, trainY)
dataset_test = SimpleDataset(testX, testY)

class NNet(nn.Module):
    def __init__(self, depth, width, act=nn.Tanh()):
        
        super(NNet, self).__init__()
 
        layers_list = []

        for l in range(depth):
            layers_list.append(nn.Linear(in_features=width, out_features=width))
            layers_list.append(act)
            
        #
        
        # last layer
        layers_list.append(nn.Linear(in_features=width, out_features = 1))

        for layer in layers_list:
            if type(layer) != nn.Tanh and type(layer) != nn.ReLU:
                if type(act) == nn.Tanh:
                    nn.init.xavier_uniform_(layer.weight)
                else:
                    nn.init.kaiming_normal_(layer.weight)
        
        # containers: https://pytorch.org/docs/stable/nn.html#containers
        self.net = nn.ModuleList(layers_list)
        
    def forward(self, X):
        self.net(X)
        # h = X
        # for layer in self.net:
        #     h = layer(h)
        # #
        # return h

#send to gpu for CUDA speeds
variableNN = NNet(3,5).to(device)

# do training 
train_loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

