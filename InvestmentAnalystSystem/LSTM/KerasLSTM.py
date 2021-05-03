from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization,Dropout
import numpy as np
#from sklearn import datasets
import keras
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from Data.UseData import read_data
import pandas as pd
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler



def get_data():

    obj_read=read_data()
    status,msg,res=obj_read.get_daily_data(codelist=['000001.SZ'],start_date=20150101,end_date=20210101,distance=1,columns=['close','rate_of_increase_next_1'])
    
    if not status:
        
        raise (msg)
    else:
        len_=len(res[0])
        len_=int(len_*0.9)
        x_train=res[0][0:len_,0:1]
        y_train=res[0][0:len_,1]
        y_train=np.where(y_train>0,1,0)
        x_test=res[0][len_:len(res[0]),0:1]
        y_test=res[0][len_:len(res[0]),1]
        y_test=np.where(y_test>0,1,0)
        return x_train,y_train,x_test,y_test

def stock_NN_v1():
    
    #需要之前90次的数据来预测下一次的数据
    need_num = 90
    #训练数据的大小
    #training_num=1000
    #迭代10次

    epoch = 10
    batch_size = 20
    sc = MinMaxScaler(feature_range=(0, 1))

    #x_train_F,y_train_F,x_test,y_test_F=get_data()
    obj_read=read_data()
    status,msg,res=obj_read.get_daily_data(codelist=['000001.SZ'],start_date=20150101,end_date=20200101,distance=1,columns=['close'])
    
    if not status:
        
        raise (msg)
    dataset=res[0]
    
    training_dataset = dataset

    training_dataset_scaled = sc.fit_transform(X=training_dataset)
     
    x_train = []
    y_train = []
    #每 need_num个连续日期后 下一个作为y，等于比如90个作为x ，第91作为y
    for i in range(need_num, training_dataset_scaled.shape[0]):
        x_train.append(training_dataset_scaled[i-need_num: i])
        y_train.append(training_dataset_scaled[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    #因为LSTM要求输入的数据格式为三维的，[training_number, time_steps, 1]，因此对数据进行相应转化
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))



    model=model_2(x_train,y_train,epoch,batch_size)
    
    status,msg,res=obj_read.get_daily_data(codelist=['000001.SZ'],start_date=20200101,end_date=20210301,distance=1,columns=['close'])
    
    if not status:
        
        raise (msg)
    dataset=res[0]
    
    

    inputs = dataset
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(X=inputs)



    x_validation = []
     
    for i in range(need_num, inputs.shape[0]):
        x_validation.append(inputs[i - need_num:i, 0])
     
    x_validation = np.array(x_validation)
    x_validation = np.reshape(x_validation, (x_validation.shape[0], x_validation.shape[1], 1))
    real_stock_price = dataset[need_num:]
    predictes_stock_price = model.predict(x=x_validation)
    predictes_stock_price = sc.inverse_transform(X=predictes_stock_price)
    print(len(predictes_stock_price))
    print(len(real_stock_price))

    plt.plot(real_stock_price, color='red', label='Real Stock Price')
    plt.plot(predictes_stock_price, color='blue', label='Predicted Stock Price')
    plt.xlabel(xlabel='Time')
    plt.legend()
    plt.show()



def model_2(x_train,y_train,epoch,batch_size):


    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(loss='mean_squared_error', optimizer='adam')
    regressor.fit(x_train, y_train, epochs = epoch, batch_size = batch_size)
    return regressor

if __name__ == '__main__':
    stock_NN_v1()
    
