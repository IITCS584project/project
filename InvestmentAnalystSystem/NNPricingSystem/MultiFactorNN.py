import torch
import torch.nn as nn
import numpy as np
class MultiFactorNN(nn.Module):
    def __init__(self, factor_num :int):
        super(MultiFactorNN, self).__init__()
        self.mFactorLayer = factor_num
        self.Init(factor_num)
        pass

    def Init(self, feature_num):
        self.mFactorLayer = nn.Sequential(
            nn.Linear(feature_num, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1 )
        )
        
        #self.mFactorLayer.apply(self.InitWeights)

    def InitWeights(self, m):
        if type(m) == nn.Linear:
            #m.weight.data.fill_(1.0)
            print(m.weight)
    
    def forward(self, x):
        return self.mFactorLayer(x)
