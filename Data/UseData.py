#!/usr/bin/python
# -*- coding: UTF-8 -*-


import time
import datetime
import sys
import csv
import json
import pandas as pd
import os
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
path=os.path.split(os.path.realpath(__file__))[0]
#print(path)

class read_data():

    def __init__(self):
        self.path=path+'/data_csv'


    def get_daily_data(self,code,start_date='',end_date='',distance=1):
        
        # 读标的物列表
        # param  code  必填，标的物编码  e.g. 000001.SZ
        # param  code  不必填，开始时间，没填写的时候，直到取最早，开始结束都没填写的时候取全部
        # param  code  不必填，结束时间，没填写的时候，直到取最晚，开始结束都没填写的时候取全部
        code=str(code.upper())
        df=pd.read_csv(self.path+"/"+code+".csv", sep=",")
        df = df.loc[df["trade_date"] >= start_date] if start_date != '' else df
        df = df.loc[df["trade_date"] <= end_date] if end_date != '' else df # 截取日期片
        df=df.sort_values(["trade_date"],ascending=True) #重新排序
        df=df.reset_index() # 重新定义索引
        
        for i in df.index:
            if i%distance!=0: #用index 取余 distance，余0返回，其他去掉
                df.drop(index=i,inplace=True)
              
        df['rate_of_increase']=((df['close']/df.shift(periods=1)['close'])-1)*100
        df=df[['ts_code','trade_date','open','high','low','close','change','vol','amount','rate_of_increase']]
        
        return df.values
            
    def get_stock_list(self):

        df = pd.read_csv(self.path + "/stocklist.csv", sep=",")
        return df


    def get_daily_data_clac(self,code,start_date='',end_date=''):
        #tips：因为有的指标是需要昨日数据的，所以如果是该股票开盘第一天的话，一些rate 会是 NaN
        #close_rate_of_increase=今日收盘价/昨天收盘价
        #avg_rate_of_increase=今日四项平均值/昨日四项平均值
        #open_today_rate_of_increase=今日收盘价／今日开盘价
        #middle_rate_of_increase=「1/2(今日最高+今日最低)」/「1/2(昨日最高+昨日最低)」
        #分类参数 档位{key1:value1 ,key2:value1~value2 ,key3 value3~value4,key4value4~}

        code=str(code.upper())
        df=pd.read_csv(self.path+"/"+code+".csv", sep=",")
        df=df.sort_values(["trade_date"],ascending=True)
        df['close_rate_of_increase']=(df['close']/df['pre_close']-1)*100
        df['avg_for_OHLC']=(df['close']+df['open']+df['high']+df['low'])/4
        df['avg_for_HL']=(df['high']+df['low'])/2
        df['open_today_rate_of_increase']=(df['open']/df['close']-1)*100
        df['avg_rate_of_increase']=((df['avg_for_OHLC']/df.shift(periods=1)['avg_for_OHLC'])-1)*100
        df['middle_rate_of_increase']=((df['avg_for_HL']/df.shift(periods=1)['avg_for_HL'])-1)*100
        del df['avg_for_OHLC']
        del df['avg_for_HL']
        df = df.loc[df["trade_date"] >= start_date] if start_date != '' else df
        df = df.loc[df["trade_date"] <= end_date] if end_date != '' else df
        return df
        return df




if __name__ == '__main__':

    #code=['sh','sz','hs300','sz50','zxb','cyb']

    obj_read=read_data() #声明对象
    result=obj_read.get_daily_data('sh',start_date=20210405,end_date=20210412,distance=1)
    # 按照开始时间和结束时间取 000001.SZ这个标的物,distance=n n代表返回日期间隔
    # 返回第1天一定是>=start_date的第一个交易日的日期数据
    # 按照distance的间隔，返回所有交易日内的间隔日期数据

    print(result)




