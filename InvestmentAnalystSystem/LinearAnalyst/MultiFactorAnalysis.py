from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
from InvestmentAnalystSystem.LinearAnalyst.FactorLinearNN import FactorLinearNN
import torch
import torch.nn as nn

class MultiFactorAnalysis:
    def __init__(self):
        pass

    def Fit(self):
        pass


def Main():
    solver = NNRegressionSystem()
    network = FactorLinearNN(3)
    loss_func = nn.MSELoss()

    pass