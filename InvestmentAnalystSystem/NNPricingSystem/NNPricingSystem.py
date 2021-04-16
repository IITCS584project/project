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


class NNPricingModel:
    def __init__(self):
        # 
        pass

    def LoadData(self, asset_tickerlist):
        reader = read_data()
        
        pass

    def Fit(self):
        model = RiskPremiaNNPricingModel()


        solver = NNRegressionSystem()
        solver.Init()
        pass


    