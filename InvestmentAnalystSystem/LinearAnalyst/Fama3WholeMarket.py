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
    stock_ticker = '600859.SH'
    start_date = 20190305
    due_date = 20200410
    #start_date = 20190201
    #due_date = 20190415
    reader = read_data()
    succ,info,factor_data = reader.get_factor_daily(start_date=start_date,end_date=due_date,columns=['trade_date', 'mkt_rf','smb','hml', 'rf'])
    mkt_rf = factor_data[:,1] * 100
    mkt_rf = mkt_rf.reshape(len(mkt_rf),1)
    smb = factor_data[:,2] * 100
    smb = mkt_rf.reshape(len(smb),1)
    hml = factor_data[:,3] * 100
    hml = hml.reshape(len(hml),1)
    rf = factor_data[:, 4] * 100
    X = np.concatenate([mkt_rf, smb, hml], axis=1)


    # 计算开市期间所有交易的股票
    reader = read_data()
    #stock_list = reader.get_trade_cal_stock_list(start_date, due_date)
    
    r2_list = []
    #succ, info, stock_dailyield = StockDataProvider.GetMultiStockYields(stock_list, start_date, due_date)
    #reader.save_local("fama3test", stock_dailyield)
    stock_dailyield = reader.read_local("fama3test")
    stock_num = stock_dailyield.shape[0]
    for k in range(stock_num):    
        stock_yield = stock_dailyield[k, :, 2]
        
        
        y = stock_yield - rf    
        y = y.reshape(len(y),1)

        ws = MultiFactorWorkspace()
        ws.LinearRegresion(X, y)
        r2 = ws.LinearRegresion(X, y)
        r2_list.append(r2)
    r2 = np.array(r2_list)
    print("r2.mean", r2.mean(), "r2.std", r2.std())
    
if __name__ == '__main__':
    Main2()