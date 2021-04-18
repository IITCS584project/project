# -*- coding: utf-8 -*-

import numpy as np
from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
from InvestmentAnalystSystem.NNPricingSystem.RiskPremiaNNPricingModel import RiskPremiaNNPricingModel
import torch
import torch.nn as nn
import torch.optim as optim
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.DrawFunctions import DrawLinearRegression
from InvestmentAnalystSystem.NNPricingSystem.RiskPremiaNNPricinNN import RiskPremiaNNPricingNN
import matplotlib.pyplot as plt


class RiskPremiaNNPricingSystem:
    def __init__(self):        
        pass
    
    def Init(self, characteristic_num :int, asset_num :int, factor_num :int):
        self.mSolver = NNRegressionSystem()
        self.mModel = RiskPremiaNNPricingNN(characteristic_num, asset_num, factor_num)
        self.mOptimizer = optim.SGD(self.mModel.parameters(), lr=0.01)
        self.mLossFunc = nn.MSELoss()
        self.mSolver.Init(self.mModel, self.mOptimizer, self.mLossFunc )
    

    def LoadData(self, asset_tickerlist):
        reader = read_data()
        
        pass

    def Fit(self):
        model = RiskPremiaNNPricingModel()
        solver = NNRegressionSystem()
        solver.Init()
        pass


    