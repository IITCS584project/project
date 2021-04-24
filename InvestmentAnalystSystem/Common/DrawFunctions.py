import matplotlib.pyplot as plt
import numpy as np

class DrawFunctions:
    @staticmethod
    def DrawLinearRegression(plt, X,y, k, b, title, x_label, y_label):
        plt.scatter(X, y, marker='x',c='r')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        max_x = np.max(X)
        min_x = np.min(X)
        x_data = np.arange(min_x, max_x, (max_x ) / 100.0)
        y_data = x_data * k + b
        plt.plot(x_data,y_data,color=(0,0,1))
        pass
    
    @staticmethod
    def DrawLoss( plt, loss :np.array, title: str, x_label :str, y_label :str):
        loss_num :int = len(loss)
        x_data :np.array = np.arange(0, loss_num)
        y = loss
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.plot(x_data, y, color=(0,0,1))
        pass
