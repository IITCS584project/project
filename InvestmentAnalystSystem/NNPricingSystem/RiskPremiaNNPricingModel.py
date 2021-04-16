import torch
import torch.nn as nn
import numpy as np

class RiskPremiaNNPricingModel(nn.Module):
    def __init__(self, characteristic_num :int, asset_num :int, factor_num :int):
        super(NNPricingModel, self).__init__()
        # graph 1:training for betas
        # hidden layer 1: (characteristic_num, characteristic_num * 2)
        self.mBetaLayer = nn.Sequential(
            nn.Linear(characteristic_num, characteristic_num * 2)
            nn.ReLU()
            nn.Linear(characteristic_num * 2, factor_num )
        )
        # graph 2: training for factors
        self.mFactorLayer = nn.Sequential(
            nn.Linear(asset_num, factor_num)        
        )
        
        pass
        
    
    def forward(self, x):
        beta_X = x[0]
        factor_X = x[1]
        beta_output = self.mBetaLayer(beta_X)
        factor_output = self.mFactorLayer(factor_X)
        return torch.dot(beta_output, factor_output)
