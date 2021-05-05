import numpy as np
from InvestmentAnalystSystem.Common.PredictResult import PredictResultType,CalcPredictResult, CalcRiseDropPredictResult
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
from InvestmentAnalystSystem.NNPricingSystem.MultiFactorNN import MultiFactorNN
from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
import torch
import torch.nn as nn
import torch.optim as optim
from InvestmentAnalystSystem.Common.StockDataProvider import StockDataProvider
import matplotlib.pyplot as plt
from InvestmentAnalystSystem.Common.DrawFunctions import DrawFunctions
from InvestmentAnalystSystem.NNPricingSystem.MultiFeatureNNPricingSystem import MultiFeatureNNPricingSystem

def Main():
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'        
    X_train, y_train, X_test, y_test = StockDataProvider().GenStockData(stock_ticker, market_ticker, 
                20190201, 20190810, 20200308, 20200315 )

    X_train = StockDataProvider.NpArrayToTensor(X_train)
    y_train = y_train.reshape(len(y_train), 1)
    y_train = StockDataProvider.NpArrayToTensor(y_train)
    X_test = StockDataProvider.NpArrayToTensor(X_test)
    y_test = y_test.reshape(len(y_test), 1)
    y_test = StockDataProvider.NpArrayToTensor(y_test)

    solver = MultiFeatureNNPricingSystem()    
    solver.FitAndTestAccuracy(X_train,y_train, X_test, y_test)
    solver.ShowParameters(plt)

    pred_y = solver.Predict(X_train)
    accuracy = solver.Accuracy(pred_y, y_train)
    print( "Train accuracy", accuracy)
    pred_y = solver.Predict(X_test)
    accuracy = solver.Accuracy(pred_y, y_test)
    print("Test accuracy", accuracy)
    pass

Main()