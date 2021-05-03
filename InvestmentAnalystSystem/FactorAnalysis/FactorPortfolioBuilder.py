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

    def BuildSingleFactor(self, stock_codelist, except_codelist, factor_columnname, start_date, end_date, rebalance_distance):
        '''
        单变量排序后，取分值最高和分值最低的两组股票，
        构建多空对冲组合，近似作为因子收益率
        factor_columnname: 用于排序的列名
        '''
        # ['ts_code', 'trade_date', 'rate_of_increase_1', 'vol', 'rate_of_increase_7'])
        yield_columnname = "rate_of_increase_1"
        date_columnindex = 1
        yield_columnindex = 2
        factor_columnindex = 3
        need_columns =  ['ts_code', 'trade_date', yield_columnname, factor_columnname]
        reader = read_data()
        # all the stock data of the whole market
        succ, info, asset_data = reader.get_daily_data( stock_codelist, except_codelist ,start_date, end_date, 1,
                   need_columns)
        if not succ:
            return False, info
        
        factor_info = StyleFactorInfo()
        #factor_info.Init(date_columnindex, yield_columnindex)        
        # pick the portfolio
        # 因子需要定期reblance，每次rebalance之后，因子内的股票都不同
        for t in range(0, asset_data.shape[1], rebalance_distance):
            # 从第一个时刻开始构建portfolio，持续rebalance distance时间
            stocks_at_t = asset_data[:, t, :]
            splited_indices = self.SortAndSplit(stocks_at_t, factor_columnindex, 4)
            low_indices = splited_indices[0]
            high_indices = splited_indices[len(splited_indices) - 1]
            # now we need to build a long-short portfolio as a factor
            # 构造风格因子，它是一个多空组合
            portfolio :SpreadPortfolioInfo = SpreadPortfolioInfo()
            # 该portfolio的持续存在时间
            duration :int = np.minimum(rebalance_distance, asset_data.shape[1] - t)            
            portfolio.Init( t, duration, high_indices, low_indices, asset_data, yield_columnindex, date_columnindex)
            factor_info.AddPortfolio(portfolio)
            
        
        return factor_info

def Main():
    builder = FactorPortfolioBuilder()
    factor = builder.BuildSingleFactor(['600859.SH', '600519.SH', 
    '002624.SZ', '600887.SH', '600016.SH', '600030.SH', '600036.SH', '600600.SH', '300600.SZ'], [], 'pb', 20190110,20191029, 20)    
    #factor = builder.BuildSingleFactor([], [], 'pb', 20190101,20191231, 20)    
    yield_data :np.array = factor.ExportToNpArray()
    factor_yields = factor.CalculateTimeSeriesYield(0,5, 20)
    is_significant ,t, p = factor.IsTimeSeriesYieldSignificant(factor_yields)
    print( 'mean',factor_yields.mean(), 'std', factor_yields.std(ddof=1))
    print(is_significant, t, p)
    pass

Main()

