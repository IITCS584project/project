import torch
import torch.nn as nn
import numpy as np
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
        lastprint = ""
        for t in range(epoch):
            # clear gradient buffer
            self.mOptimizer.zero_grad()
            # forward pass
            y_pred = self.mModel(X)
            # calculate loss
            loss = self.mLossFunc(y_pred, y)            
            lastprint = str(t) + "\t" + str(loss.item())
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
    
    def ExtractModelParameter(self):
        
        total_params = []
        for name, parameter in self.mModel.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            print(name, param)
            total_params.append([name,param])
        
        return total_params

    def Draw(self, plt):
        #k,b = self.ExtractModelParameter()
        #DrawLinearRegression(plt, self.mMarketYield.numpy(), self.mAssetYield.numpy(), k, b, "CAPM", self.mMarketTicker, self.mAssetTicker)
        #plt.show()
        pass

