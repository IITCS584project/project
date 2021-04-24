import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn

class FactorLinearNN(nn.Module):
    def __init__(self, feature_num :int):
        super(FactorLinearNN, self).__init__()
        self.mFeatureNum = feature_num
        self.fc1 = nn.Linear(feature_num, 1)

    def forward(self, x):
        fc1 = self.fc1(x)
        return fc1

