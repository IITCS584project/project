import numpy as np
from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
from InvestmentAnalystSystem.LinearAnalyst.FactorLinearNN import FactorLinearNN
import torch
import torch.nn as nn
import torch.optim as optim
from Data.UseData import read_data
import matplotlib.pyplot as plt
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
from InvestmentAnalystSystem.LinearAnalyst.MultiFactorSystem import MultiFactorWorkspace
from InvestmentAnalystSystem.Common.StockDataProvider import StockDataProvider



def Main2():
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'
    start_date = 20190305
    due_date = 20200410
    #start_date = 20190201
    #due_date = 20190415
    reader = read_data()
    succ,info,factor_data = reader.get_factor_daily(start_date=start_date,end_date=due_date,columns=['trade_date', 'mkt_rf', 'rf'])
    mkt_rf = factor_data[:,1] * 100
    mkt_rf = mkt_rf.reshape(len(mkt_rf),1)
    rf = factor_data[:, 2] * 100
    

    # 市场因子
    date_column = 1
    yield_column = 2
    # 一期的天数
    epoch_days = 1
    #succ, info, market_dailyyield = StockDataProvider.GetStockYields(market_ticker, start_date, due_date )
    succ, info, stock_dailyield = StockDataProvider.GetStockYields(stock_ticker, start_date, due_date)
    stock_yield = stock_dailyield[:,2]
    X = mkt_rf
    y = stock_yield - rf    
    y = y.reshape(len(y),1)
    

    # 做时间序列上的回归
    # 训练期数，比如训练使用过去5期的数据，ground truth用未来1期的数据，采用mse损失函数
    train_epochs = 20
    ws = MultiFactorWorkspace()
    
    ws.Run(0.02, 1500, start_date, due_date, train_epochs, X, y)
    
Main2()