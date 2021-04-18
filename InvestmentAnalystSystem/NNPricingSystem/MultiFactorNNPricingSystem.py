import numpy as np
from InvestmentAnalystSystem.Common.PredictResult import PredictResultType,CalcPredictResult
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
from InvestmentAnalystSystem.NNPricingSystem.MultiFactorNN import MultiFactorNN
from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
import torch
import torch.nn as nn
import torch.optim as optim

class MultiFactorNNPricingSystem:
    def __init__(self):
        pass

    def Init(self, feature_num :int):
        self.mSolver = NNRegressionSystem()
        self.mModel = MultiFactorNN(feature_num)
        self.mOptimizer = optim.SGD(self.mModel.parameters(), lr=0.01)
        self.mLossFunc = nn.MSELoss()
        self.mSolver.Init(self.mModel, self.mOptimizer, self.mLossFunc )

    def LoadData(self, ticker_list, start_date, end_date, distance, columns):
        reader = read_data()
        stock_data = reader.get_daily_data(ticker_list,start_date, end_date, distance,columns)
        # the first row of yield is na
        stock_data = stock_data[:, 1:]
        return stock_data
        

    def Fit(self, X,y):
        self.Init(X.shape[1])
        self.mSolver.Fit(X, y, 2000)        
        
    def Predict(self, X):
        return self.mSolver.Predict(X)
        
    
    def Accuracy(self, pred_y, true_y):
        '''
        calculate the accuracy of the prediction y
        '''        
        return (pred_y == true_y).sum() / len(true_y)


def Main():
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'
    solver = MultiFactorNNPricingSystem()    
    market_data = solver.LoadData([market_ticker], 20190304, 20200409, 1, ['ts_code', 'trade_date','rate_of_increase'])
    stock_data = solver.LoadData([stock_ticker], 20190305, 20200410, 1, ['ts_code', 'trade_date','rate_of_increase'])
    X = market_data[0, :, 2]
    X = X.reshape((len(X),1))
    y = stock_data[0, :, 2]
    
    X = torch.from_numpy(np.array(X, dtype=float)).float()
    y = torch.from_numpy(np.array(y, dtype=float)).float()
    X_train, y_train, X_test, y_test = UtilFuncs.SplitData(X, y, 2.0 / 3.0, True)
    solver.Fit(X_train,y_train)
    pred_y = solver.Predict(X_test)
    pred_y = CalcPredictResult(pred_y)
    y_test = CalcPredictResult(y_test)
    accuracy = solver.Accuracy(pred_y, y_test)
    print("predict accuracy", accuracy)
    pass

Main()