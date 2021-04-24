import torch
import torch.nn as nn
import numpy as np
from InvestmentAnalystSystem.Common.DrawFunctions import DrawFunctions
class NNRegressionSystem:
    def __init__(self):
        self.mModel :nn.Module = None
        self.mOptimizer = None
        self.mLossFunc = None
        self.mLoss = []
        pass

    def Init( self, model :nn.Module, optimizer , loss_func):
        self.mModel = model
        self.mOptimizer = optimizer
        self.mLossFunc = loss_func
        pass

    
    def Fit(self, X, y, epoch):
        lastprint = ""

        for t in range(epoch):
            # clear gradient buffer
            self.mOptimizer.zero_grad()
            # forward pass
            y_pred = self.mModel(X)
            # calculate loss
            loss = self.mLossFunc(y_pred, y)                        
            loss_item = loss.item()
            lastprint = str(t) + "\t" + str(loss_item)  
            self.mLoss.append(loss_item)
            print(lastprint, end="")
            print("\b" * len(lastprint) * 2, end="", flush=True)
            # backward pass
            loss.backward()
            # apply the weights
            self.mOptimizer.step()
        print("")
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
    
    def ShowParameters(self):
        kv_map = self.mModel.state_dict()
        result_map = {}
        for k, v in kv_map.items():
            print(k, v.numpy())
            result_map[k] = v.numpy()
        return result_map

    def Draw(self, plt):
        #k,b = self.ExtractModelParameter()
        #DrawLinearRegression(plt, self.mMarketYield.numpy(), self.mAssetYield.numpy(), k, b, "CAPM", self.mMarketTicker, self.mAssetTicker)
        #plt.show()
        loss :np.array = np.array(self.mLoss)
        DrawFunctions.DrawLoss(plt, loss, "Loss", "iterations", "loss")
        pass

