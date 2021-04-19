from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
from InvestmentAnalystSystem.LinearAnalyst.FactorLinearNN import FactorLinearNN
import torch
import torch.nn as nn
import torch.optim as optim
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.PredictResult import PredictResultType,CalcPredictResult
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
from InvestmentAnalystSystem.Common.StockDataProvider import StockDataProvider
import numpy as np

class MultiFactorSystem:
    def __init__(self):
        pass

    def Init(self, feature_num :int):
        self.mSolver = NNRegressionSystem()
        self.mModel = FactorLinearNN(feature_num)
        self.mOptimizer = optim.SGD(self.mModel.parameters(), lr=0.001)
        self.mLossFunc = nn.MSELoss()
        self.mSolver.Init(self.mModel, self.mOptimizer, self.mLossFunc )
        pass
    
    def Fit(self, X, y):
        self.Init(X.shape[1])
        self.mSolver.Fit(X, y, 5000)
        pass

    def Predict(self, X :np.array):
        return self.mSolver.Predict(X)
        

    def Accuracy(self, pred_y :np.array, true_y :np.array):
        pred_y = CalcPredictResult(pred_y)
        true_y = CalcPredictResult(true_y)
        return (pred_y == true_y).sum() / len(true_y)

def Main():
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'
    solver = MultiFactorSystem()    
    X, y = StockDataProvider.GetStockDataForPredict(stock_ticker, market_ticker, 20200305, 20200410)
    X = StockDataProvider.NpArrayToTensor(X)
    y = StockDataProvider.NpArrayToTensor(y)
    X_train, y_train, X_test, y_test = UtilFuncs.SplitData(X, y, 2.0 / 3.0, True)
    solver.Fit(X_train,y_train)
    pred_y = solver.Predict(X_test)
    accuracy = solver.Accuracy(pred_y, y_test)
    print(accuracy)
    pass

Main()