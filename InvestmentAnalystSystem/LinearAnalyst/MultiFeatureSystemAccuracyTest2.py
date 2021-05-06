from InvestmentAnalystSystem.Common.StockDataProvider import StockDataProvider
from Data.UseData import read_data
from InvestmentAnalystSystem.LinearAnalyst.MultiFeatureSystem import MultiFeatureSystem

def Main(code,train_start,train_end,test_start,test_end):
    market_ticker = 'hs300'
    stock_ticker = code   
    X_train, y_train, X_test, y_test = StockDataProvider().GenStockData(stock_ticker, market_ticker, 
                train_start, train_end, test_start, test_end )
    
    X_train = StockDataProvider.NpArrayToTensor(X_train)
    y_train = y_train.reshape(len(y_train), 1)
    y_train = StockDataProvider.NpArrayToTensor(y_train)
    X_test = StockDataProvider.NpArrayToTensor(X_test)
    y_test = y_test.reshape(len(y_test), 1)
    y_test = StockDataProvider.NpArrayToTensor(y_test)

    solver = MultiFeatureSystem()    
    solver.SetDefaultLR(0.0001)
    solver.Fit(X_train,y_train, 100000)
    solver.ShowParameters()

    pred_y = solver.Predict(X_train)
    accuracy = solver.Accuracy(pred_y, y_train)
    print( "Train accuracy", accuracy)
    pred_y = solver.Predict(X_test)
    accuracy = solver.Accuracy(pred_y, y_test)
    print("Test accuracy", accuracy)


if __name__ == '__main__':
    obj_read=read_data()
    #list_=obj_read.get_trade_cal_stock_list(start_date=20190101,end_date=20210201)
    #list_=list_[0:100]
    #print(len(list_))
    #list_=['000001.SZ','000002.SZ']
    train_start=20200602
    train_end=20210101
    test_start=20210101
    test_end=20210201
    code = '000023.SZ'
    Main(code=code,train_start=train_start,train_end=train_end,test_start=test_start,test_end=test_end)