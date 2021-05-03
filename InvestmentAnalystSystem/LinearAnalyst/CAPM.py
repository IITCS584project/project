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
    
Main2()