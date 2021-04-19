from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
from InvestmentAnalystSystem.LinearAnalyst.FactorLinearNN import FactorLinearNN
import torch
import torch.nn as nn
import torch.optim as optim
from Data.UseData import read_data
class MultiFactorSystem:
    def __init__(self):
        pass

    def Init(self, feature_num :int):
        self.mSolver = NNRegressionSystem()
        self.mModel = FactorLinearNN(feature_num)
        self.mOptimizer = optim.SGD(self.mModel.parameters(), lr=0.01)
        self.mLossFunc = nn.MSELoss()
        self.mSolver.Init(self.mModel, self.mOptimizer, self.mLossFunc )
        pass

    def LoadData( self, code_list, begin_date, end_date, distance):
        '''
        anallyze by CAPM
        e.g. market_ticker='hs300', asset_ticker='002624.sz'
        begin_date = 20200305
        end_date = 20200412
        '''
        reader = read_data()        
        yield_list = reader.get_daily_data(code_list,start_date=begin_date,end_date=end_date,distance=distance,columns=['ts_code', 'trade_date', 'rate_of_increase'])
        # the first row of yield is na
        yield_list = yield_list[:, 1:, :]
        return yield_list

    def Fit(self, X, y):
        self.mSolver.Fit(X, y, 500)
        pass