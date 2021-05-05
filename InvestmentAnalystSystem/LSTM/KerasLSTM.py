from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization,Dropout
import numpy as np
#from sklearn import datasets
import keras
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
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

def stock_NN_v1(code,train_start,train_end,test_start,test_end):
    
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
    status,msg,res=obj_read.get_daily_data(codelist=[code],start_date=train_start,end_date=train_end,distance=1,columns=['close'])
    
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
    
    status,msg,res=obj_read.get_daily_data(codelist=[code],start_date=test_start,end_date=test_end,distance=1,columns=['close'])
    
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
    #print(predictes_stock_price)
    #print(real_stock_price)
    predict__=[]
    real__=[]
    for r in predictes_stock_price:
        predict__.append(r[0])
    for k in real_stock_price:
        real__.append(k[0])
    #print(predict__,real__)
    y_predict=[]
    for i in range(1,len(predict__)):
        if predict__[i]>predict__[i-1]:
            y_predict.append(1)
        else:
            y_predict.append(0)
    y_real=[]
    for i in range(1,len(real__)):
        if real__[i]>real__[i-1]:
            y_real.append(1)
        else:
            y_real.append(0)
    acc=0
    for i in range(0,len(y_predict)):
        if y_predict[i]==y_real[i]:
            acc+=1

    print(acc/len(y_predict))
    return acc/len(y_predict)
    #print(len(y_predict),len(y_real))
    # plt.plot(real_stock_price, color='red', label='Real Stock Price')
    # plt.plot(predictes_stock_price, color='blue', label='Predicted Stock Price')
    # plt.xlabel(xlabel='Time')
    # plt.legend()
    # plt.show()



def stock_NN_v2():
    
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
     
    # for i in range(need_num, inputs.shape[0]):
    #     x_validation.append(inputs[i - need_num:i, 0])
    x_validation.append(inputs[0:90, 0])
    
    predict_res=[]
    for r in range(0,189):
        
        x_validation = np.array(x_validation)
        x_validation = np.reshape(x_validation, (x_validation.shape[0], x_validation.shape[1], 1))
        predictes_stock_price = model.predict(x=x_validation)
        predict_res.append(sc.inverse_transform(X=predictes_stock_price)[0])
        x=[]
        x=x_validation[0][1:90,0].tolist()
        x.append(predictes_stock_price[0][0])
        x=np.array(x)
        x_validation = []
        x_validation.append(x)
        
    real_stock_price = dataset[need_num:]
    print(predict_res)
    print(real_stock_price)

    plt.plot(real_stock_price, color='red', label='Real Stock Price')
    plt.plot(predict_res, color='blue', label='Predicted Stock Price')
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
    regressor.fit(x_train, y_train, epochs = epoch, batch_size = batch_size,verbose=0)
    return regressor

if __name__ == '__main__':
    
    obj_read=read_data()
    list_=obj_read.get_trade_cal_stock_list(start_date=20190101,end_date=20210201)
    result={}
    i=0
    for code in list_:
        i+=1
        tmp=stock_NN_v1(code=code,train_start=20190101,train_end=20200101,test_start=20200101,test_end=20210201)
        result[code]=tmp
        print(code,tmp,i,len(list_))
    print(result)