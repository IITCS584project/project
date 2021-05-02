from InvestmentAnalystSystem.FactorAnalysis.SpreadPortfolioInfo import SpreadPortfolioInfo
import numpy as np
from scipy import stats
class StyleFactorInfo:
    def __init__(self):
        self.mPortfolios = []
        pass

    def Init(self, date_column, yield_column):
        self.mDateColumn = date_column
        self.mYieldColumn = yield_column

    def AddPortfolio(self, portfolio :SpreadPortfolioInfo):
        self.mPortfolios.append(portfolio)
        pass
    
    

    def CalculateFactorYield(self, start_tindex, end_tindex):
        '''
        计算因子某一段时间内的总收益率
        
        start_date : 开始时间
        end_date : 结束时间
        distance : 
        '''
        # calculate the average return in the time series
        # 首先计算一个时间序列内的总收益率
        portfolio : SpreadPortfolioInfo = None
        # 时间序列的收益率        
        total_yield = 1.0
        for portfolio in self.mPortfolios:
                if not portfolio.IsValidTime(start_tindex) and not portfolio.IsValidTime(end_tindex):
                    continue
                portfolio_yield = portfolio.CalculateYield(start_tindex, end_tindex)                
                total_yield *= 1.0 + portfolio_yield
        return total_yield - 1.0
    
    def CalculateTimeSeriesYield(self, begin_tindex :int, distance :int, epoch_num :int) -> np.array:
        '''
        计算从begin_tindex开始，间隔为distance，连续epoch_num期的时间序列收益
        '''
        factor_yields :np.array = np.zeros(epoch_num)
        cur_index = 0
        for t in range(begin_tindex, begin_tindex + distance * epoch_num, distance):
            begin_tindex = t
            end_tindex = t + distance - 1
            factor_yield :float = self.CalculateFactorYield(begin_tindex, end_tindex)
            factor_yields[cur_index] = factor_yield
            cur_index = cur_index + 1

        return factor_yields


    def IsTimeSeriesYieldSignificant(self, ts_yield :np.array):
        '''
        当前时间序列的收益率是否显著不为0
        '''
        result = stats.ttest_1samp(ts_yield, 0.0)
        t = result.statistic
        p = result.pvalue
        # 1.65,1.96,2.58
        # 90%, 95%, 99%
        if t > 1.65:
            return True, t, p
        else:
            return False, t, p

    