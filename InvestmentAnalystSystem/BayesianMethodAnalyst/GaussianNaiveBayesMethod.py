import numpy as np
from sklearn.naive_bayes import GaussianNB
from InvestmentAnalystSystem.Common.PredictResult import PredictResultType,CalcRiseDropPredictResult
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
from InvestmentAnalystSystem.Common.StockDataProvider import StockDataProvider
class GaussianNaiveBayesMethod:
    def __init__(self):
        self.mClf = None
        pass
    
    def LoadData(self, ticker_list, start_date, end_date, distance, columns):
        reader = read_data()
        stock_data = reader.get_daily_data(ticker_list,start_date, end_date, distance,columns)
        # the first row of yield is na
        stock_data = stock_data[:, 1:]
        return stock_data
    
    def Fit(self, X, y):
        self.mClf = GaussianNB()
        #X = X.reshape((len(X),1))
        self.mClf.fit(X,y)

    def Predict(self, X):
        #X = X.reshape((len(X),1))
        pred_y = self.mClf.predict(X)
        return pred_y

    def Accuracy(self, pred_y, true_y):
        '''
        calculate the accuracy of the prediction y
        '''        
        return (pred_y == true_y).sum() / len(true_y)

def Main():
    solver = GaussianNaiveBayesMethod()
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'        
    X_train, y_train, X_test, y_test = StockDataProvider().GenStockData(stock_ticker, market_ticker, 
                20190401, 20190810, 20190901, 20191001 )
    y_train = CalcRiseDropPredictResult(y_train)
    y_test = CalcRiseDropPredictResult(y_test)
    '''
    X_train, y_train = StockDataProvider.GetStockDataForPredict(stock_ticker, market_ticker, 20190401, 20190810)
    y_train = y_train.reshape(len(y_train))
    y_train = CalcRiseDropPredictResult(y_train)

    X_test, y_test = StockDataProvider.GetStockDataForPredict(stock_ticker, market_ticker, 20190901, 20191001)
    y_test = y_test.reshape(len(y_test))
    y_test = CalcRiseDropPredictResult(y_test)
    '''
    
    solver.Fit(X_train,y_train)
    pred_y = solver.Predict(X_test)
    
    accuracy = solver.Accuracy(pred_y, y_test)
    print(accuracy)    
    pass

Main()