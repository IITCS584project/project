from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
from InvestmentAnalystSystem.LinearAnalyst.FactorLinearNN import FactorLinearNN
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.DrawFunctions import DrawLinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def Main():
    solver = NNRegressionSystem()
    model = FactorLinearNN(1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_func = nn.MSELoss()
    solver.Init(model, optimizer, loss_func )
    

    reader = read_data()
    hs300= reader.get_daily_data('hs300',start_date=20200305,end_date=20200412,distance=1,columns=['ts_code','trade_date','rate_of_increase'])
    wmsj= reader.get_daily_data('002624.sz',start_date=20200305,end_date=20200412,distance=1,columns=['ts_code','trade_date','rate_of_increase'])

    hs300_increaserate = hs300[1:, 2]
    wmsj_increaserate = wmsj[1:,2]
    hs300_increaserate = (hs300_increaserate - np.mean(hs300_increaserate)) / np.std(hs300_increaserate)
    wmsj_increaserate = (wmsj_increaserate - np.mean(wmsj_increaserate)) / np.std(wmsj_increaserate)
    hs300_ir = torch.tensor(np.array(hs300_increaserate, dtype=float)).float()
    wmsj_ir = torch.tensor(np.array(wmsj_increaserate, dtype=float)).float()
    hs300_ir = hs300_ir.reshape((len(hs300_ir), 1))
    wmsj_ir = wmsj_ir.reshape((len(wmsj_ir),1))
    
    print(hs300_ir)
    print(wmsj_ir)
    solver.Fit(hs300_ir, wmsj_ir,200)
    print("weights:")
    args = [0,0]
    param_idx = 0
    for name, param in model.named_parameters():
        print(name, param.item())
        args[param_idx] = param.item()
        param_idx += 1 

    DrawLinearRegression(plt, hs300_ir.numpy(), wmsj_ir.numpy(), args[0], args[1], "CAPM", "HS300", "Perfect World")
    plt.show()
    
    #print(hs300_increaserate)
    #print(wmsj_increaserate)

    #solver.Fit()
    
    
    pass

Main()