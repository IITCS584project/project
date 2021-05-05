from InvestmentAnalystSystem.NNPricingSystem.MultiFeatureNNPricingSystem import MultiFeatureNNPricingSystem
from InvestmentAnalystSystem.Common.StockDataProvider import StockDataProvider
import matplotlib.pyplot as plt
import numpy as np
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
def Main():
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'        
    
    train_epochs = 30  
    loss = []
    # 行是时间
    # X：股票的参数
    # y: 下一期的涨幅
    X, y = StockDataProvider.GetStockDataForTTest(stock_ticker, market_ticker, 20190201, 20190415 )                
    itr_num = len(y) - (train_epochs + 1)
    for k in range(itr_num):
        # 从k期到k+train_epochs-1期为训练数据
        X_train = X[k:k+train_epochs, :]
        y_train = y[k:k+train_epochs]
        X_train = StockDataProvider.NpArrayToTensor(X_train)            
        y_train = StockDataProvider.NpArrayToTensor(y_train)
        # k+train_epochs期为测试数据
        X_test = X[k + train_epochs, :]
        y_test = y[k + train_epochs]
        X_test = StockDataProvider.NpArrayToTensor(X_test)            
        X_test = X_test.reshape(1, len(X_test))
        y_test = StockDataProvider.NpArrayToTensor(y_test)
        y_test = y_test.reshape(1,1)

        solver = MultiFeatureNNPricingSystem()    
        solver.Fit(X_train, y_train)
        pred_y = solver.Predict(X_test)
        loss.append(pred_y - y_test)
    loss = np.array(loss)
    is_significant, t, p = UtilFuncs.IsSignificant(loss)
        
    print("is significant:", is_significant, "t:", t, "p:", p, "loss.std:", loss.std(ddof = 1), "loss.mean:", loss.mean())

    
    

Main()