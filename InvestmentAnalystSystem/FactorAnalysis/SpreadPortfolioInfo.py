# -*- coding: UTF-8 -*-
import numpy as np
class SpreadPortfolioInfo:
    '''
    差价组合，就是多空组合
    固定
    '''
    def __init__(self):
        pass
    def Init(self, begintime_index, duration, long_stockindices, short_stockindices, wholemarket_stocks, yieldcolumnindex, datecolumnindex):        
        # 在全部stock中的时间开始索引
        self.mBeginTimeIndex = begintime_index
        # 当前portfolio持续多少个交易日
        self.mDuration = duration
        # 做多的股票列表索引
        self.mLongIndices = long_stockindices
        # 做空的股票列表索引
        self.mShortIndices = short_stockindices
        # 全市场股票信息
        self.mMarketStocks = wholemarket_stocks
        # 收益率对应列
        self.mYieldColumnIndex = yieldcolumnindex
        # 日期对应列
        self.mDateColumnIndex = datecolumnindex
        # long nav
        self.mLongNAV :np.array = None
        # short nav
        self.mShortNAV :np.array = None
        # 每日收益率
        self.mDailyYield :np.array = None

        self.CalculateDailyNAV(self.mYieldColumnIndex)

    def GetDateList(self):
        date_list :np.array = self.mMarketStocks[0, self.mBeginTimeIndex : self.mBeginTimeIndex + self.mDuration, self.mDateColumnIndex ]
        return date_list

    def ExportToNPArray(self):
        date_list = self.GetDateList()
        # 按列求和，就是所有股票净值相加
        longnav_bydate = self.mLongNAV.sum(axis=0)
        shortnav_bydate = self.mShortNAV.sum(axis=0)
        nav_bydate = longnav_bydate + shortnav_bydate
        nav_yesterday = nav_bydate[:-1]
        nav_yesterday = np.concatenate([np.ones(len(nav_bydate),1), nav_yesterday])
        yield_bydate = (nav_bydate - nav_yesterday) / nav_yesterday
        # 转置为列向量
        date_list = date_list.reshape(len(date_list),1)
        yield_bydate = nav_bydate.reshape(len(yield_bydate),1)
        # 两列连接        
        dat = np.concatenate([date_list, yield_bydate], axis=1)        
        return dat


    def IsValidTime(self, tindex):
        return tindex >= self.mBeginTimeIndex and tindex < self.mBeginTimeIndex + self.mDuration

    def CalculateDailyNAV(self, yield_column):
        '''
        计算每日净值
        '''
        #(long_num, days_num, feature_num)
        long_stocks = self.mMarketStocks[self.mLongIndices, :, :]
        long_num = len(self.mLongIndices)
        #(short_num, days_num, feature_num)
        short_stocks = self.mMarketStocks[self.mShortIndices, :, :]
        short_num = len(self.mShortIndices)
        #(stock_num, times)
        self.mLongNAV = np.zeros((long_stocks.shape[0], self.mDuration))
        self.mShortNAV = np.zeros((short_stocks.shape[0], self.mDuration))
        # 建仓日第初始净值为
        long_nav = np.full(long_stocks.shape[0], 0.5 / long_num)
        short_nav = np.full(short_stocks.shape[0], 0.5/short_num)
        # 假设是首日开盘建仓        
        for t in range(self.mBeginTimeIndex, self.mBeginTimeIndex + self.mDuration):
            # 第t日
            #(stock_num, )
            long_yield = long_stocks[:,t, yield_column]
            short_yield = short_stocks[:, t, yield_column]
            # 计算当天收盘时的nav
            long_nav = long_nav * (long_yield * 0.01 + 1)
            short_nav = short_nav * (-short_yield * 0.01 + 1)
            self.mLongNAV[:, t - self.mBeginTimeIndex] = long_nav
            self.mShortNAV[:, t - self.mBeginTimeIndex] = short_nav

        pass
    
    def CalculateYield(self, start_tindex, end_tindex):
        '''
        计算开始日期到结束日期的总收益率
        '''
        if (start_tindex >= self.mBeginTimeIndex + self.mDuration) or (end_tindex < self.mBeginTimeIndex):
            # 不在时间范围内，不会对yield产生变化
            return 0.0
        
        if start_tindex <= self.mBeginTimeIndex:
            start_tindex = self.mBeginTimeIndex
        if end_tindex >= self.mBeginTimeIndex + self.mDuration:
            end_tindex = self.mBeginTimeIndex + self.mDuration - 1
        
        if end_tindex < start_tindex:
            return 0.0

        # 统计从开始日到结束日到总收益率
        # 实际上是从开始日前一天计算净值，到结束日时到净值差异
        long_startnav = 0.5
        short_startnav = 0.5
        if start_tindex > self.mBeginTimeIndex:
            # 开始日前一天的nav
            long_startnav = self.mLongNAV[:, start_tindex - 1 - self.mBeginTimeIndex]
            short_startnav = self.mShortNAV[:, start_tindex - 1 - self.mBeginTimeIndex]
            # 所有股票相加
            long_startnav = long_startnav.sum()
            short_startnav = short_startnav.sum()

        
        
        long_endnav = self.mLongNAV[:, end_tindex - self.mBeginTimeIndex].sum()
        short_endnav = self.mShortNAV[:, end_tindex - self.mBeginTimeIndex].sum()

        start_nav = long_startnav + short_startnav
        end_nav = long_endnav + short_endnav
        ret = (end_nav - start_nav ) / start_nav
        return ret
