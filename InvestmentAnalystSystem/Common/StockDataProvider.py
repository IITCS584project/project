from Data.UseData import read_data
import numpy as np

class StockDataProvider:
    @staticmethod
    def GetStockDataForPredict( asset_ticker, market_ticker, start_date, end_date ):
        reader = read_data()
        today_data = reader.get_daily_data( [asset_ticker] ,start_date, end_date, 1,['ts_code', 'trade_date', 'rate_of_increase'])
        yesterday_data = reader.get_daily_data( [asset_ticker] ,start_date, end_date, 1,['vol'])
        # remove 1st row 
        today_data = today_data[1:]
        yesterday_data = yesterday_data[:-1,:]
        stock_data = np.concatenate([today_data, yesterday_data], axis=1)
        return stock_data
    