# -*- coding: UTF-8 -*-
from Data.UseData import read_data
from InvestmentAnalystSystem.FactorAnalysis.SpreadPortfolioInfo import SpreadPortfolioInfo
from InvestmentAnalystSystem.FactorAnalysis.StyleFactorInfo import StyleFactorInfo
import numpy as np
class FactorPortfolioBuilder:
    def __init__(self):
        pass

    
    def SortAndSplit(self, stock_array: np.array, sort_column: int, split_num :int):
        '''
        单变量排序，将所有股票按照某个变量排序，取分值最高和分值最低的两组股票，        
        stock_array: (stock_num, feature_num)
        '''
        # we sort the stock list by the factor variable
        indices :np.array = stock_array[:, sort_column].argsort()
        splited_indices = np.array_split(indices, split_num)
        return splited_indices

    
    def CalculateFama3Stocks(self, stock_array :np.array, pb_column :int, totalmv_column):
        #pb从小到大排
        pb_lh_indices = stock_array[:, pb_column].argsort()
        #市值从小到达排
        mv_lh_indices = stock_array[:, totalmv_column].argsort()
        #市值按1:1分块
        low_mv, high_mv = np.array_split(mv_lh_indices, 2)
        #PB按3:4:3分块
        low_pb, med_pb, high_pb =  np.array_split(pb_lh_indices, [3,4,3])
        return low_mv, high_mv, low_pb, high_pb

    def BuildFama3(self, start_date, end_date, rebalance_distance):
        date_column = 1
        pb_column = 2
        totalmv_column = 3
        yield_column = 4
        need_columns =  ['ts_code', 'trade_date', 'pb', 'total_mv', 'rate_of_increase_1']
        loaded_stockinfo, trading_cal = self.LoadStockData(need_columns, start_date, end_date, rebalance_distance)
        # 遍历每个时间区间，构造因子portfolio
        smb_info = StyleFactorInfo()
        hml_info = StyleFactorInfo()
        smb_info.SetTradingCalendar(trading_cal)
        hml_info.SetTradingCalendar(trading_cal)
        for stock_info in loaded_stockinfo:
            # 遍历时间区间
            asset_data, begin_date, due_date = stock_info
            stocks_at_t = asset_data[:,0,:]
            low_mv, high_mv, low_pb, high_pb = self.CalculateFama3Stocks(stocks_at_t, pb_column, totalmv_column )

            # 该portfolio的持续存在期数
            duration :int = asset_data.shape[1]
            # now we need to build a long-short portfolio as a factor
            # 构造smb的portfolio
            portfolio :SpreadPortfolioInfo = SpreadPortfolioInfo()
            portfolio.Init( begin_date, due_date, duration, low_mv, high_mv, asset_data, yield_column, date_column)
            smb_info.AddPortfolio(portfolio)
            # 构造hml的portfolio
            portfolio :SpreadPortfolioInfo = SpreadPortfolioInfo()
            portfolio.Init(begin_date, due_date, duration, low_pb, high_pb, asset_data, yield_column, date_column)
            hml_info.AddPortfolio(portfolio)
        return smb_info, hml_info

    def LoadStockData(self, need_columns, start_date, end_date, rebalance_distance):
        reader = read_data()

        # 首先分段，某个时间区间下，哪些股票可用全部提取出来        
        available_stocks = []
        df_trade_cal=reader.read_trade_cal(start_date=start_date,end_date=end_date)
        df_trade_cal['trade_date']=df_trade_cal['cal_date']
        df_trade_cal=df_trade_cal[df_trade_cal['is_open']==1]
        # 交易日列表
        trading_cal = df_trade_cal['trade_date'].to_numpy()
        for k in range(0, len(trading_cal), rebalance_distance):
            # 建仓阶段：首先把每个rebalance周期内有效股票拿出来
            begin_date = trading_cal[k]
            due_index = k + rebalance_distance -1 
            due_index = np.minimum(due_index, len(trading_cal) - 1)
            due_date = trading_cal[due_index]
            
            stock_list = reader.get_trade_cal_stock_list(begin_date, due_date)
            available_stocks.append((stock_list, begin_date, due_date))
        loaded_stockinfo = []
        # 加载每个时间区间内的股票信息
        for stock_info in available_stocks:            
            stock_list, begin_date, due_date = stock_info
            succ, info, asset_data = reader.get_daily_data( stock_list, [] ,begin_date, due_date, 1,
                   need_columns)
            if not succ:
                print("Fail!")
                print(info)
                return
            loaded_stockinfo.append((asset_data, begin_date, due_date))
        return loaded_stockinfo, trading_cal
        


    def BuildSingleFactor(self, stock_codelist, except_codelist, factor_columnname, start_date, end_date, rebalance_distance, long_high = True):
        '''
        单变量排序后，取分值最高和分值最低的两组股票，
        构建多空对冲组合，近似作为因子收益率
        factor_columnname: 用于排序的列名
        rebalance_distance: 多少个交易日后重新构建一次
        long_high: 做多评分高的组合
        '''
        # ['ts_code', 'trade_date', 'rate_of_increase_1', 'vol', 'rate_of_increase_7'])
        yield_columnname = "rate_of_increase_1"
        date_columnindex = 1
        yield_columnindex = 2
        factor_columnindex = 3
        need_columns =  ['ts_code', 'trade_date', yield_columnname, factor_columnname]
        reader = read_data()

        # 首先分段，某个时间区间下，哪些股票可用全部提取出来
        
        available_stocks = []
        df_trade_cal=reader.read_trade_cal(start_date=start_date,end_date=end_date)
        df_trade_cal['trade_date']=df_trade_cal['cal_date']
        df_trade_cal=df_trade_cal[df_trade_cal['is_open']==1]
        # 交易日列表
        trading_cal = df_trade_cal['trade_date'].to_numpy()
        '''
        for k in range(0, len(trading_cal), rebalance_distance):
            # 建仓阶段：首先把每个rebalance周期内有效股票拿出来
            begin_date = trading_cal[k]
            due_index = k + rebalance_distance -1 
            due_index = np.minimum(due_index, len(trading_cal) - 1)
            due_date = trading_cal[due_index]
            
            stock_list = reader.get_trade_cal_stock_list(begin_date, due_date)
            available_stocks.append((stock_list, begin_date, due_date))
        '''
        for k in range(0, len(trading_cal), rebalance_distance):
            # 建仓阶段：首先把每个rebalance周期内有效股票拿出来
            begin_date = trading_cal[k]
            due_index = k + rebalance_distance -1 
            due_index = np.minimum(due_index, len(trading_cal) - 1)
            due_date = trading_cal[due_index]
            
            stock_list = stock_codelist
            available_stocks.append((stock_list, begin_date, due_date))
        
        
        loaded_stockinfo = []
        # 加载每个时间区间内的股票信息
        for stock_info in available_stocks:            
            stock_list, begin_date, due_date = stock_info
            succ, info, asset_data = reader.get_daily_data( stock_list, [] ,begin_date, due_date, 1,
                   need_columns)
            if not succ:
                print("Fail!")
                print(info)
                return
            loaded_stockinfo.append((asset_data, begin_date, due_date))
        # 把所有股票一共分为几段   
        split_blocks :int = 2
        # 遍历每个时间区间，构造因子portfolio
        factor_info = StyleFactorInfo()
        factor_info.SetTradingCalendar(trading_cal)
        for stock_info in loaded_stockinfo:
            # 遍历时间区间
            asset_data, begin_date, due_date = stock_info
            stocks_at_t = asset_data[:,0,:]
            splited_indices = self.SortAndSplit(stocks_at_t, factor_columnindex, split_blocks )
            low_indices = splited_indices[0]
            high_indices = splited_indices[len(splited_indices) - 1]
            # now we need to build a long-short portfolio as a factor
            # 构造风格因子，它是一个多空组合
            portfolio :SpreadPortfolioInfo = SpreadPortfolioInfo()
            # 该portfolio的持续存在期数
            duration :int = asset_data.shape[1]
            long_indices = None
            short_indices = None
            if long_high:
                # 做多评分高的组合
                long_indices = high_indices
                short_indices = low_indices
            else:
                long_indices = low_indices
                short_indices = high_indices

            portfolio.Init( begin_date, due_date, duration, long_indices, short_indices, asset_data, yield_columnindex, date_columnindex)
            factor_info.AddPortfolio(portfolio)
        
        return factor_info

def Main():
    builder = FactorPortfolioBuilder()
    #factor = builder.BuildSingleFactor(['600859.SH', '600519.SH', 
    #'002624.SZ', '600887.SH', '600016.SH', '600030.SH', '600036.SH', '600600.SH', '300600.SZ'], [], 'pb', 20190110,20191029, 20)    

    #factor = builder.BuildSingleFactor(['600859.SH', '600519.SH', 
    #'002624.SZ', '600887.SH', '600016.SH', '600030.SH', '600036.SH', '600600.SH', '300600.SZ'], [], 'pb', 20190110,20190310, 20, False)    

    #factor = builder.BuildSingleFactor(['600859.SH', '600519.SH'], [], 'pb', 20190110,20190310, 20)    

    #factor = builder.BuildSingleFactor([], [], 'pb', 20190101,20191231, 20)    
    smb, hml = builder.BuildFama3( 20190110, 20190310, 20 )
    factor = smb
    yield_data :np.array = factor.ExportToNpArray()
    reader = read_data()
    reader.save_local("PBLMH", yield_data)
    yield_data2 = reader.read_local("PBLMH")
    factor_yields = factor.CalculateTimeSeriesYield(20190110,5, 20)
    is_significant ,t, p = factor.IsTimeSeriesYieldSignificant(factor_yields)
    print( 'mean',factor_yields.mean(), 'std', factor_yields.std(ddof=1))
    print(is_significant, t, p)
    pass

Main()

