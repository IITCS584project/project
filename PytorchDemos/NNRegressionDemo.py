
from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
from InvestmentAnalystSystem.LinearAnalyst.FactorLinearNN import FactorLinearNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TorchNN(nn.Module):
    def __init__(self):
        super(TorchNN, self).__init__()
        # hidden layer 1: 2 input, 10 neurons(output)
        self.fc1 = nn.Linear(3, 10)
        self.s1 = nn.Sigmoid()
        # hidden layer 2: 10 input,1 neuron(output)
        self.fc2 = nn.Linear(10, 1)
        self.s2 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.s1(x)
        x = self.fc2(x)
        x = self.s2(x)
        return x

def GenSamples(feature_cnt, sample_cnt):
    # feature_cnt-dimension vector, if the sample is in the sphere with radius 1, then it is class 1, else class 2
    X = np.random.rand(sample_cnt, feature_cnt)
    y = np.zeros(shape=(sample_cnt, 1))
    for k in range(sample_cnt):
        y[k] = np.dot( X[k], X[k] ) <= 1
    #y = X.mean(axis=1)
    #y[y>=0.5] = 1
    #y[y<0.5] = 0    
    return torch.tensor(X).float(), torch.tensor(y).float()

def Split(X, y):
    row_cnt = X.shape[0]
    split_point = int(row_cnt * 2 / 3)
    X_train = X[:split_point]
    X_test = X[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]
    return X_train, y_train, X_test, y_test

def Accuracy(pred, true_y):
    data_num :int = len(true_y)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    correct_num = (pred == true_y).sum().item()
    correct_num /= data_num
    print(correct_num)

def Main():
    X, y = GenSamples(3, 2000)
    X_train, y_train, X_test, y_test = Split(X,y)
    solver = NNRegressionSystem()
    model = TorchNN()
    loss_func = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1)
    solver.Init(model, optimizer, loss_func)
    solver.Fit(X_train, y_train, 10000)
    pred_y = solver.Predict(X_train)
    Accuracy(pred_y, y_train)

    pass

Main()