import torch
import torch.nn as nn
class NNRegressionSystem:
    def __init__(self):
        self.mModel :nn.Module = None
        self.mOptimizer = None
        self.mLossFunc = None
        pass

    def Init( self, model :nn.Module, optimizer , loss_func):
        self.mModel = model
        self.mOptimizer = optimizer
        self.mLossFunc = loss_func
        pass

    
    def Fit(self, X, y, epoch):
        for t in range(epoch):
            # clear gradient buffer
            self.mOptimizer.zero_grad()
            # forward pass
            y_pred = self.mModel(X)
            # calculate loss
            loss = self.mLossFunc(y_pred, y)
            print(t, loss.item())
            # backward pass
            loss.backward()
            # apply the weights
            self.mOptimizer.step()
        pass

    def Predict(self, X):
        with torch.no_grad():
            # forward pass
            pred = self.mModel(X)
            return pred

    def Loss(self, pred, true_y):
        # calculate loss
        loss = self.mLossFunc(pred, true_y)
        # accumulate loss
        total_loss = loss.item()
        return total_loss
    

