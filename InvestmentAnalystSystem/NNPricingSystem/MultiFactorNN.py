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
            nn.Linear(feature_num, feature_num * 2),
            nn.ReLU(),
            nn.Linear(feature_num * 2, feature_num * 2),
            nn.ReLU(),
            nn.Linear(feature_num * 2, 1 )
        )
    
    def forward(self, x):
        return self.mFactorLayer(x)
