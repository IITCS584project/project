import torch
from torch import nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

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

def SimpleSample():
    X = np.array([0.38,0.86])
    y = np.copy(X)
    y[y>=0.5] = 1
    y[y < 0.5] = 0
    return X,y

class TorchNN(nn.Module):
    def __init__(self):
        super(TorchNN, self).__init__()
        # hidden layer 1: 2 input, 10 neurons(output)
        self.fc1 = nn.Linear(2, 10)
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


def DoTrain( X, y, model, optimizer, loss_func, epoch):
    for t in range(epoch):
        # clear gradient buffer
        optimizer.zero_grad()
        # forward pass
        y_pred = model(X)
        # calculate loss
        loss = loss_func(y_pred, y)
        print(t, loss.item())
        # backward pass
        loss.backward()
        # apply the weights
        optimizer.step()
    pass

def DoTest(X, true_y, model, loss_func):
    total_loss = 0.0
    data_num = len(true_y)
    # the test changes nothing
    with torch.no_grad():
        # forward pass
        pred = model(X)
        # calculate loss
        loss = loss_func(pred, true_y)
        # accumulate loss
        total_loss += loss.item()
        # calculate the correct ratio
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        correct_num = (pred == true_y).sum().item()
        correct_num /= data_num
        print("correct", correct_num)
    pass

def Main():
    x_train, y_train = GenSamples(2, 3000)
    x_test, y_test = GenSamples(2, 100)

    model = TorchNN()
    model = model.to("cpu")
    print(model)
    
    loss_func = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1)
    DoTrain(x_train, y_train, model, optimizer, loss_func, 15000)
    DoTest(x_test, y_test, model, loss_func)

Main()