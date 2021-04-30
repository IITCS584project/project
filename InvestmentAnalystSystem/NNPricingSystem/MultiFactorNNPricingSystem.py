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

class MultiFactorNNPricingSystem:
    def __init__(self):
        pass

    def Init(self, feature_num :int):
        self.mSolver = NNRegressionSystem()
        self.mModel = MultiFactorNN(feature_num)
        self.mOptimizer = optim.SGD(self.mModel.parameters(), lr=0.0004, momentum=0.9, weight_decay = 0.5)
        #self.mOptimizer = optim.Adam(self.mModel.parameters(), lr=0.0004, weight_decay=0)
        self.mLossFunc = nn.MSELoss()
        self.mSolver.Init(self.mModel, self.mOptimizer, self.mLossFunc )
        
        

    def Fit(self, X,y):
        self.Init(X.shape[1])
        self.mSolver.Fit(X, y, 50000)        
        
    def Predict(self, X):
        return self.mSolver.Predict(X)
        
    
    def Accuracy(self, pred_y, true_y):
        '''
        calculate the accuracy of the prediction y
        '''        
        pred_y = CalcRiseDropPredictResult(pred_y)
        true_y = CalcRiseDropPredictResult(true_y)
        return (pred_y == true_y).sum() / len(true_y)

    def ShowParameters(self, plt):
        self.mSolver.ShowParameters()
        self.mSolver.Draw(plt)
        plt.show()


def Main():
    
    '''
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'
    solver = MultiFactorNNPricingSystem()    
    X, y = StockDataProvider.GetStockDataForPredict(stock_ticker, market_ticker, 20200401, 20200810)    
    X = StockDataProvider.NpArrayToTensor(X)
    y = StockDataProvider.NpArrayToTensor(y)
    X_train, y_train, X_test, y_test = UtilFuncs.SplitData(X, y, 2.0 / 3.0, True)
    '''
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'        
    X_train, y_train, X_test, y_test = StockDataProvider().GenStockData(stock_ticker, market_ticker, 
                20190401, 20190810, 20190820, 20190920 )

    X_train = StockDataProvider.NpArrayToTensor(X_train)
    y_train = y_train.reshape(len(y_train), 1)
    y_train = StockDataProvider.NpArrayToTensor(y_train)
    X_test = StockDataProvider.NpArrayToTensor(X_test)
    y_test = y_test.reshape(len(y_test), 1)
    y_test = StockDataProvider.NpArrayToTensor(y_test)

    solver = MultiFactorNNPricingSystem()    
    solver.Fit(X_train,y_train)
    solver.ShowParameters(plt)

    pred_y = solver.Predict(X_test)
    accuracy = solver.Accuracy(pred_y, y_test)
    print("accuracy", accuracy)
    pass
'''
def Main2():
    X, y = StockDataProvider.DummyGenerateStockData(100)
    solver = MultiFactorNNPricingSystem()
    X = StockDataProvider.NpArrayToTensor(X)
    y = StockDataProvider.NpArrayToTensor(y)
    X_train, y_train, X_test, y_test = UtilFuncs.SplitData(X, y, 2.0 / 3.0, True)
    solver.Fit(X_train,y_train)
    solver.ShowParameters(plt)
    pred_y = solver.Predict(X_test)
    accuracy = solver.Accuracy(pred_y, y_test)
    print(accuracy)
'''

Main()