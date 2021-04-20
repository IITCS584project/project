import numpy as np
from InvestmentAnalystSystem.Common.PredictResult import PredictResultType,CalcPredictResult
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
from InvestmentAnalystSystem.NNPricingSystem.MultiFactorNN import MultiFactorNN
from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
import torch
import torch.nn as nn
import torch.optim as optim
from InvestmentAnalystSystem.Common.StockDataProvider import StockDataProvider

class MultiFactorNNPricingSystem:
    def __init__(self):
        pass

    def Init(self, feature_num :int):
        self.mSolver = NNRegressionSystem()
        self.mModel = MultiFactorNN(feature_num)
        self.mOptimizer = optim.SGD(self.mModel.parameters(), lr=0.1, momentum=0.9)
        self.mLossFunc = nn.MSELoss()
        self.mSolver.Init(self.mModel, self.mOptimizer, self.mLossFunc )
        

    def Fit(self, X,y):
        self.Init(X.shape[1])
        self.mSolver.Fit(X, y, 5000)        
        
    def Predict(self, X):
        return self.mSolver.Predict(X)
        
    
    def Accuracy(self, pred_y, true_y):
        '''
        calculate the accuracy of the prediction y
        '''        
        pred_y = CalcPredictResult(pred_y)
        true_y = CalcPredictResult(true_y)
        return (pred_y == true_y).sum() / len(true_y)


def Main():
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'
    solver = MultiFactorNNPricingSystem()    
    X, y = StockDataProvider.GetStockDataForPredict(stock_ticker, market_ticker, 20190305, 20200410)
    X = StockDataProvider.NpArrayToTensor(X)
    y = StockDataProvider.NpArrayToTensor(y)
    X_train, y_train, X_test, y_test = UtilFuncs.SplitData(X, y, 2.0 / 3.0, True)
    solver.Fit(X_train,y_train)
    pred_y = solver.Predict(X_train)
    accuracy = solver.Accuracy(pred_y.T, y_train)
    print(accuracy)
    pass

def Main2():
    X, y = StockDataProvider.DummyGenerateStockData(100)
    solver = MultiFactorNNPricingSystem()
    X = StockDataProvider.NpArrayToTensor(X)
    y = StockDataProvider.NpArrayToTensor(y)
    X_train, y_train, X_test, y_test = UtilFuncs.SplitData(X, y, 2.0 / 3.0, True)
    solver.Fit(X_train,y_train)
    
    pred_y = solver.Predict(X_train)
    accuracy = solver.Accuracy(pred_y.T, y_train)
    print(accuracy)

Main2()