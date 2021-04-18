# -*- coding: utf-8 -*-

import numpy as np
from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
from InvestmentAnalystSystem.NNPricingSystem.RiskPremiaNNPricingModel import RiskPremiaNNPricingModel
import torch
import torch.nn as nn
import torch.optim as optim
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.DrawFunctions import DrawLinearRegression
import matplotlib.pyplot as plt


class RiskPremiaNNPricingSystem:
    def __init__(self):
        self.mSolver = NNRegressionSystem()
        self.mModel = FactorLinearNN(1)
        self.mOptimizer = optim.SGD(self.mModel.parameters(), lr=0.01)
        self.mLossFunc = nn.MSELoss()
        self.mSolver.Init(self.mModel, self.mOptimizer, self.mLossFunc )
        pass

    def LoadData(self, asset_tickerlist):
        reader = read_data()
        
        pass

    def Fit(self):
        model = RiskPremiaNNPricingModel()
        solver = NNRegressionSystem()
        solver.Init()
        pass


    