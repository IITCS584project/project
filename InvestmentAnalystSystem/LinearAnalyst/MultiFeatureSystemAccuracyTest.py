from InvestmentAnalystSystem.LinearAnalyst.MultiFeatureSystem import MultiFeatureSystem
from InvestmentAnalystSystem.Common.StockDataProvider import StockDataProvider
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
from InvestmentAnalystSystem.LinearAnalyst.MultiFeatureSystem import MultiFeatureSystem
import matplotlib.pyplot as plt
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

    solver = MultiFeatureSystem()    
    solver.SetDefaultLR(0.0035)
    solver.Fit(X_train,y_train, 100000)
    solver.ShowParameters()

    pred_y = solver.Predict(X_train)
    accuracy = solver.Accuracy(pred_y, y_train)
    print( "Train accuracy", accuracy)
    pred_y = solver.Predict(X_test)
    accuracy = solver.Accuracy(pred_y, y_test)
    print("Test accuracy", accuracy)
    pass

Main()