from InvestmentAnalystSystem.FactorLinearAnalyst import FactorLinearNN
import pytorch.nn as nn
import torch.optim as optim

class LinearRegressionNN:
    def __init__(self, feature_num :int):
        super(LinearRegressionNN, self).__init__()
        self.mFeatureNum = feature_num
        self.fc1 = nn.Linear(feature_num, 1)
    def forward(self, x):
        x = self.fc1(x)
        return x

class FactorLinearAnslysis:
    def __init__(self):
        pass

    def Fit(self, X, y, model, optimizer, loss_func, epoch):
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

