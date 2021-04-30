# -*- coding: UTF-8 -*-
class SpreadPortfolioInfo:
    '''
    差价组合，就是多空组合
    固定
    '''
    def __init__(self):
        pass
    def Init(self, begin_date, end_date, begintime_index, long_stockindices, short_stockindices, wholemarket_stocks, yieldcolumnindex):
        self.mBeginDate = begin_date
        self.mEndDate = end_date
        # 在全部stock中的时间开始索引
        self.mBeginTimeIndex = begintime_index
        self.mLongIndices = long_stockindices
        self.mShortIndices = short_stockindices
        self.mMarketStocks = wholemarket_stocks
        self.mYieldColumnIndex = yieldcolumnindex

    def IsValidTime(self, date):
        return date >= self.mBeginDate and date <= self.mEndDate

    def ConvertDateToTimeIndex(self, date_column :int, date :int):
        date_index = np.where(long_stocks[:, date_column, :] == date) 
        return date_index

        
    def ConvertTimeIndexToDate(self, date_column :int, time_index :int):
        return self.mMarketStocks[0, time_index, date_column]

    def CalculateYieldEqualWeighted(self, start_date, end_date, date_column, yield_column) ->float:
        # assuming it is equal weighted portfolio
        # 等权重portfolio
        if start_date < self.mBeginDate:
            start_date = self.mBeginDate
        if end_date > self.mEndDate:
            end_date = self.mEndDate
        
        if end_date < start_date:
            return False

        # 多头股票
        long_stocks = self.mMarketStocks[self.mLongIndices, :]
        time_startindex = self.ConvertDateToTimeIndex(date_column, start_date)
        time_endindex = self.ConvertDateToTimeIndex(date_column, end_date)

        long_num = long_stocks.shape[0]
        
        # 空头股票
        short_stocks = self.mMarketStocks[self.mShortIndices, :]
        short_num = short_stocks.shape[0]
        # 由于我们是在期初重新reblance这个投资组合，这个投资组合还是等权重的，
        # 所以算中间某一段时间的yield就要从期初开始构建

        # 期初多头空头等金额
        weight_long = np.full((long_num,), 0.5 / long_num)
        weight_short = np.full((short_num,), 0.5 / short_num)
        
        epochs = 0
        #从构建组合前到开始时间前的时间加权收益率        
        long_yields_before = long_stocks[:, self.mBeginTimeIndex: time_startindex, yield_column]
        long_yields_before = (long_yields_before * 0.01 + 1).prod(axis=1)
        
        short_yields_before = short_stocks[:,self.mBeginTimeIndex: time_startindex, yield_column]
        short_yields_before = (short_yields_before * 0.01 + 1).prod(axis=1)

        # 从构建组合到开始时间前的净值
        netval_before = np.dot(weight_long, long_yields_before) - np.dot(weight_short, short_yields_before)

        #从构建组合到结束时间的时间加权收益率
        long_yields_toend = long_stocks[:, self.mBeginTimeIndex: time_endindex + 1, yield_column]
        long_yields_toend = (long_yields_toend * 0.01 + 1).prod(axis=1)

        short_yields_toend = short_stocks[:,self.mBeginTimeIndex: time_endindex + 1, yield_column]
        short_yields_toend = ( short_yields_toend * 0.01 + 1).prod(axis=1)

        # 从构建组合到结束时间的净值
        netval_toend = np.dot(weight_long, long_yields_toend) - np.dot(weight_short, short_yields_toend)
        final_yield = (netval_toend - netval_before) / netval_before
        return final_yield

