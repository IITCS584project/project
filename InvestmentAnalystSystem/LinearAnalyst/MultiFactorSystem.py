from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
from InvestmentAnalystSystem.LinearAnalyst.FactorLinearNN import FactorLinearNN
import torch
import torch.nn as nn
import torch.optim as optim
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.PredictResult import PredictResultType,CalcPredictResult, CalcRiseDropPredictResult, JustifyRiseDrop
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
from InvestmentAnalystSystem.Common.StockDataProvider import StockDataProvider
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
import numpy as np

class MultiFactorSystem:
    def __init__(self):
        self.mLR = 0.02
        self.mEpochs = 1500
        pass

    def SetFitParameters(self, lr, epochs):
        self.mLR = lr
        self.mEpochs = epochs

    def Init(self, feature_num :int, lr = 0.02):
        self.mSolver = NNRegressionSystem()
        self.mModel = FactorLinearNN(feature_num)
        self.mOptimizer = optim.SGD(self.mModel.parameters(), lr=lr)
        self.mLossFunc = nn.MSELoss()
        self.mSolver.Init(self.mModel, self.mOptimizer, self.mLossFunc )
        pass
    
    def Fit(self, X, y):
        self.Init(X.shape[1], self.mLR)
        self.mSolver.Fit(X, y, self.mEpochs)
        predict_y = self.mSolver.Predict(X)
        r2 = UtilFuncs.R2(predict_y, y)
        adj_r2 = UtilFuncs().AdjustedR2(X.shape[1], predict_y, y)
        return r2, adj_r2

    def Predict(self, X :torch.Tensor):
        return self.mSolver.Predict(X)
        

    def Accuracy(self, pred_y :np.array, true_y :np.array):
        pred_y = CalcPredictResult(pred_y)
        true_y = CalcPredictResult(true_y)
        return (pred_y == true_y).sum() / len(true_y)

    def ShowParameters(self):
        self.mSolver.ShowParameters()


class MultiFactorWorkspace:
    def __init__(self):

        pass

    def Run(self, lr, epochs, start_date, due_date, train_epochs, X, y):
        X = StockDataProvider.NpArrayToTensor(X)
        y = StockDataProvider.NpArrayToTensor(y)
        solver = MultiFactorSystem()
        solver.SetFitParameters(lr, epochs)
        loss = []
        r2_list = []
        adjr2_list = []
        accurate_num  = 0
        total_testnum = 0
        # 构造训练样本，每次训练样本数量为train_epochs个，预测样本为后面1期
        for k in range(0, len(y) - 1 - train_epochs):
            # 每行是一个时间样本，注意用于训练的x和y都是当期的
            train_X = X[k:k+ train_epochs, :]
            train_y = y[k : k + train_epochs]
            # 用于测试的y是下一期的            
            test_y = y[k + train_epochs]

            train_X = StockDataProvider.NpArrayToTensor(train_X)            
            train_y = StockDataProvider.NpArrayToTensor(train_y)
            r2, adj_r2 = solver.Fit(train_X, train_y)
            r2_list.append(r2.item())
            adjr2_list.append(adj_r2.item())
            
            # 训练集各因子过去各期的均值作为该因子的下一期的预期收益
            # 按列取均值
            train_X_mean = train_X.mean(axis=0)
            
            predict_y = solver.Predict(train_X_mean)
            predict_y = predict_y.numpy()[0]
            
            cur_loss = test_y - predict_y
            predict_risedrop = JustifyRiseDrop(predict_y)
            test_risedrop = JustifyRiseDrop(test_y)
            total_testnum += 1
            if predict_risedrop == test_risedrop:
                accurate_num += 1            
            loss.append(cur_loss.item())
        loss = np.array(loss)
        r2_list = np.array(r2_list)
        adjr2_list = np.array(adjr2_list)
        is_significant, t, p = UtilFuncs.IsSignificant(loss)
        
        print("is significant:", is_significant, "t:", t, "p:", p, "loss.std:", loss.std(ddof = 1), 
                    "loss.mean:", loss.mean(), 'avg r2', r2_list.mean(), 'avg adj r2', adjr2_list.mean())
        print( "accuracy", accurate_num * 1.0 / total_testnum )

        pass
'''
def Main2():
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'
    start_date = 20190305
    due_date = 20200410

    # 市场因子
    date_column = 1
    yield_column = 2
    # 一期的天数
    epoch_days = 1
    succ, info, market_dailyyield = StockDataProvider.GetStockYields(market_ticker, start_date, due_date )
    succ, info, stock_dailyield = StockDataProvider.GetStockYields(stock_ticker, start_date, due_date)
    rf = 3.0
    market_yield = UtilFuncs.TransformDailyYieldWithEpoch(market_dailyyield, start_date, due_date, date_column, yield_column, epoch_days)
    stock_yield = UtilFuncs.TransformDailyYieldWithEpoch(stock_dailyield, start_date, due_date, date_column, yield_column, epoch_days)
    market_yield -= rf
    stock_yield -= rf
    market_yield = market_yield.reshape(len(market_yield),1)
    stock_yield = stock_yield.reshape(len(stock_yield), 1)

    # 做时间序列上的回归
    # 训练期数，比如训练使用过去5期的数据，ground truth用未来1期的数据，采用mse损失函数
    train_epochs = 20
    ws = MultiFactorWorkspace()
    ws.Run(start_date, due_date, train_epochs, market_yield, stock_yield)


def Main():
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'
    start_date = 20190305
    due_date = 20200410
    

    # 这里输入的就不是股票的feature了，而是因子收益率
    # 规模因子
    smb :np.array = None    
    # 质量因子
    hml : np.array = None
    
    # 市场因子
    date_column = 1
    yield_column = 2
    # 一期的天数
    epoch_days = 1
    succ, info, market_dailyyield = StockDataProvider.GetStockYields(market_ticker, start_date, due_date )
    succ, info, stock_dailyield = StockDataProvider.GetStockYields(stock_ticker, start_date, due_date)

    #smb_yield :np.array  = smb[1]
    #hml_yield :np.array = hml[1]
    market_yield = UtilFuncs.TransformDailyYieldWithEpoch(market_dailyyield, start_date, due_date, date_column, yield_column, epoch_days)
    stock_yield = UtilFuncs.TransformDailyYieldWithEpoch(stock_dailyield, start_date, due_date, date_column, yield_column, epoch_days)    

    # 做时间序列上的回归
    # 训练期数，比如训练使用过去5期的数据，ground truth用未来1期的数据，采用mse损失函数
    train_epochs = 20

    solver = MultiFactorSystem()
    # 注意，训练的时候，都是当期训练
    X = market_yield
    y = stock_yield
    
    X = StockDataProvider.NpArrayToTensor(X)
    y = StockDataProvider.NpArrayToTensor(y)

    loss = []
    rf = 3.0
    # 构造训练样本，每次训练样本数量为train_epochs个，预测样本为后面1期
    for k in range(0, len(market_yield) - 1 - train_epochs):
        market_train = market_yield[k:k+ train_epochs] - rf
        train_y = stock_yield[k : k + train_epochs] - rf
        market_test = market_yield[k + train_epochs]
        test_y = stock_yield[k + train_epochs] - rf

        market_train = market_train.reshape(len(market_train),1)
        market_train = StockDataProvider.NpArrayToTensor(market_train)
        train_y = train_y.reshape(len(train_y), 1)
        train_y = StockDataProvider.NpArrayToTensor(train_y)
        solver.Fit(market_train, train_y)
        # 训练集各因子过去各期的均值作为该因子的下一期的预期收益
        market_train_mean = market_train.mean()
        market_train_mean = market_train_mean.reshape(1,1)
        predict_y = solver.Predict(market_train_mean)
        predict_y = predict_y.numpy()[0,0]
        cur_loss = test_y - predict_y
        loss.append(cur_loss)
    loss = np.array(loss)
    is_significant, t, p = UtilFuncs.IsSignificant(loss)
    
    print("is significant:", is_significant, "t:", t, "p:", p, "loss.std:", loss.std(ddof = 1), "loss.mean:", loss.mean())
    
    #X_train, y_train, X_test, y_test = UtilFuncs.SplitData(X, y, 2.0 / 3.0, True)
    #solver.Fit(X_train,y_train)
    #solver.ShowParameters()
    #pred_y = solver.Predict(X_test)
    #accuracy = solver.Accuracy(pred_y.T, y_test)
    #print(accuracy)
    pass


Main2()
'''