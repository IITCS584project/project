from InvestmentAnalystSystem.Common.NNRegressionSystem import NNRegressionSystem
from InvestmentAnalystSystem.LinearAnalyst.FactorLinearNN import FactorLinearNN
from Data.UseData import read_data
import torch
import torch.nn as nn
import torch.optim as optim



def Main():
    solver = NNRegressionSystem()
    model = FactorLinearNN(1)
    optimizer = optim.SGD(model.parameters(), lr=1)
    loss_func = nn.MSELoss()
    solver.Init(model, optimizer, loss_func )
    

    reader = read_data()
    result=reader.get_daily_data('sh',start_date=20210405,end_date=20210412,distance=1)
    print(result)

    #solver.Fit()
    
    
    pass

Main()