from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#from sklearn import datasets
import keras
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from Data.UseData import read_data
import pandas as pd
from keras import optimizers
 
def test():
    # X,y=load_dataset()
    x_train, y_train, x_test, y_test=get_data()
    y_train=y_train.reshape(len(y_train),1)
    
    y_train=keras.utils.to_categorical(y_train, num_classes=2)
    #print(y_train)
    y_test=y_test.reshape(len(y_test),1)
    y_test=keras.utils.to_categorical(y_test, num_classes=2)
    model = Sequential()
    #print(y_train)
    
    model.add(Dense(units=10, activation='relu',input_shape=x_train.shape))
    model.add(Dense(units=2, activation='softmax'))
    sgd = optimizers.SGD(lr=0.5, decay=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    res=model.fit(x_train, y_train, epochs=100, batch_size=1)
    res=res.history
    #loss_and_metrics = model.evaluate(x_test, y_test, batch_size=1)
    #print(loss_and_metrics)
    



    plt.plot(res['loss'],label='loss')
    plt.plot(res['accuracy'],label='accuracy') 
    plt.legend()
    plt.show()

def get_data():

    obj_read=read_data()
    status,msg,res=obj_read.get_daily_data(codelist=['000001.SZ'],start_date=20200101,end_date=20210101,distance=1,columns=['pe','pb','rate_of_increase_next_1'])
    if not status:
        
        raise (msg)
    else:
        x_train=res[0][0:200,0:2]
        y_train=res[0][0:200,2]
        y_train=np.where(y_train>0,1,0)
        x_test=res[0][200:243,0:2]
        y_test=res[0][200:243,2]
        y_test=np.where(y_test>0,1,0)
        return x_train,y_train,x_test,y_test
if __name__ == '__main__':
    test()

    