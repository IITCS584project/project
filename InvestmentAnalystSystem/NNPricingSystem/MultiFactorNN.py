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
            nn.Linear(feature_num, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1 )
        )
    
    def forward(self, x):
        return self.mFactorLayer(x)
