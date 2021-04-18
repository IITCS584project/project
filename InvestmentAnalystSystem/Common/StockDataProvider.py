from Data.UseData import read_data
import numpy as np

class StockDataProvider:
    @staticmethod
    def GetStockData( asset_ticker, market_ticker, start_date, end_date, distance ):
        reader = read_data()
        asset_data = reader.get_daily_data( [asset_ticker] ,start_date, end_date, distance,['rate_of_increase','vol'])
        # the first row of yield is na
        stock_data = asset_data[:, 1:]
        pass