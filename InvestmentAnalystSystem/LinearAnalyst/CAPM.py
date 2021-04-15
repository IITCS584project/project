import numpy as np
from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
from InvestmentAnalystSystem.LinearAnalyst.FactorLinearNN import FactorLinearNN
import torch
import torch.nn as nn
import torch.optim as optim
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.DrawFunctions import DrawLinearRegression
import matplotlib.pyplot as plt
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs

class CAPM:
    def __init__(self):
        pass
    
    def Init(self):
        self.mSolver = NNRegressionSystem()
        self.mModel = FactorLinearNN(1)
        self.mOptimizer = optim.SGD(self.mModel.parameters(), lr=0.01)
        self.mLossFunc = nn.MSELoss()
        self.mSolver.Init(self.mModel, self.mOptimizer, self.mLossFunc )
        pass

    def LoadData( self, asset_ticker, begin_date, end_date, distance):
        '''
        anallyze by CAPM
        e.g. market_ticker='hs300', asset_ticker='002624.sz'
        begin_date = 20200305
        end_date = 20200412
        '''
        reader = read_data()
        #market_yield = reader.get_daily_data(market_ticker,start_date=begin_date,end_date=end_date,distance=distance,columns=['ts_code','trade_date','rate_of_increase'])
        #asset_yield = reader.get_daily_data(asset_ticker,start_date=begin_date,end_date=end_date,distance=distance,columns=['ts_code','trade_date','rate_of_increase'])
        asset_yield = reader.get_daily_data(asset_ticker,start_date=begin_date,end_date=end_date,distance=distance,columns=['ts_code', 'trade_date', 'rate_of_increase'])
        # the first row is na
        asset_yield = asset_yield[1:, 2]
        return asset_yield

    def Fit(self, rf, market_ticker, market_yield :np.array, asset_ticker, asset_yield :np.array):
        '''
        fit the model
        rf: risk free rate
        '''
        self.Init()

        self.mMarketTicker = market_ticker
        self.mAssetTicker = asset_ticker
        
        # get the excess yield
        market_yield = market_yield - rf
        asset_yield = asset_yield - rf
        #normalzie the market yield and asset yield
        #market_yield = (market_yield - np.mean(market_yield)) / np.std(market_yield)
        #asset_yield = (asset_yield - np.mean(asset_yield)) / np.std(asset_yield)
        # change them to torch tensor
        market_yield = torch.tensor(np.array(market_yield, dtype=float)).float()
        asset_yield = torch.tensor(np.array(asset_yield, dtype=float)).float()
        market_yield = market_yield.reshape((len(market_yield), 1))
        asset_yield = asset_yield.reshape((len(asset_yield),1))

        self.mMarketYield = market_yield
        self.mAssetYield = asset_yield

        self.mSolver.Fit(market_yield, asset_yield,2000)
        pass

    def Summary(self):
        # statistical summary of the CAPM
        pred_assetyield = self.mSolver.Predict(self.mMarketYield)
        market_yield = self.mMarketYield.numpy()
        pred_assetyield = pred_assetyield.numpy()
        r2 = UtilFuncs.R2(pred_assetyield, market_yield)
        print("R2", r2)
        pass
    
    def ExtractModelParameter(self):
        args = [0,0]
        param_idx = 0
        for name, param in self.mModel.named_parameters():
            print(name, param.item())
            args[param_idx] = param.item()
            param_idx += 1 
        return args[0], args[1]

    def Draw(self, plt):
        k,b = self.ExtractModelParameter()
        DrawLinearRegression(plt, self.mMarketYield.numpy(), self.mAssetYield.numpy(), k, b, "CAPM", self.mMarketTicker, self.mAssetTicker)
        plt.show()


def Main():
    asset_ticker = '600859.sh'
    market_ticker = 'hs300'
    capm = CAPM()
    asset_yield :np.array = capm.LoadData(asset_ticker, 20190305, 20200412, 5 )
    market_yield :np.array = capm.LoadData(market_ticker, 20190305, 20200412, 5)
    capm.Fit(3, market_ticker, market_yield, asset_ticker, asset_yield)
    capm.Draw(plt)
    plt.show()
    capm.Summary()

Main()
    