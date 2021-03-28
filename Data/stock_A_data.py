#!/usr/bin/python
# -*- coding: UTF-8 -*-


import time
import datetime
import sys
import csv
import json
import tushare as ts
import pandas as pd
import requests
import os
ts.set_token('6b57e4c863aa1e2d55c8ce62742130dd38332aafe1c43f2666dedfe5')
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
pro = ts.pro_api()
path=os.path.split(os.path.realpath(__file__))[0]
#print(path)
class get_data_from_tushare():

    def __init__(self):
        self.pro = ts.pro_api()
        self.data=''
        self.path=path+'/data_csv'
    def get_stock_list(self):
        #获取标的物的列表
        self.data = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        self.data.to_csv(self.path+'/stocklist.csv')
    def write_detail_data(self):
        #从tushare按照标的物列表逐个写入日数据
        for i in range(0,self.data.shape[0]):
            ts_code = self.data.iloc[i]['ts_code']
            df = pro.daily(ts_code=ts_code)
            df.to_csv(self.path+'/' + ts_code + '.csv')
            time.sleep(1) # 对方接口要求控制频次，每分钟不能超过500次，文件有大有小，是有概率超的，所以统一sleep下
            del (ts_code)
    def write_detail_data_minutes(self,pagesize):
        #从sina按照标的物列表逐个写入5分钟数据
        #对方有爬虫限制，目前暂时看每次减缓10s实验中
        for i in range(0,self.data.shape[0]):
            ts_code = self.data.iloc[i]['ts_code']
            ts_code=ts_code.split('.')[1]+ts_code.split('.')[0]
            request_url='http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol='+ts_code+'&scale=5&ma=no&datalen='+str(pagesize)
            response = requests.get(request_url)
            print(response.content)
            res = json.loads(response.content)

            time.sleep(10)
            df = pd.DataFrame(res)
            df.to_csv(self.path+'/minutes/' + ts_code + 'minute.csv')
            del(df)

class read_data():

    def __init__(self):
        self.path=path+'/data_csv'


    def get_daily_data(self,code,start_date='',end_date=''):
        
        # 读标的物列表
        # param  code  必填，标的物编码  e.g. 000001.SZ
        # param  code  不必填，开始时间，没填写的时候，直到取最早，开始结束都没填写的时候取全部
        # param  code  不必填，结束时间，没填写的时候，直到取最晚，开始结束都没填写的时候取全部
        code=str(code.upper())
        df=pd.read_csv(self.path+"/"+code+".csv", sep=",")
        df = df.loc[df["trade_date"] >= start_date] if start_date != '' else df
        df = df.loc[df["trade_date"] <= end_date] if end_date != '' else df
        df=df.sort_values(["trade_date"],ascending=True)
        return df
    def get_stock_list(self):

        df = pd.read_csv(self.path + "/stocklist.csv", sep=",")
        return df
if __name__ == '__main__':


    obj_read=read_data()
    test=obj_read.get_stock_list()
    #return dataframe  ts_code  symbol      name area industry  list_date
    test=obj_read.get_daily_data('000001.SZ',start_date=20210301,end_date=20210302)
    # # param  code  必填，标的物编码  e.g. 000001.SZ
    # # param  code  不必填，开始时间，没填写的时候，直到取最早，开始结束都没填写的时候取全部,格式为8位数字
    # # param  code  不必填，结束时间，没填写的时候，直到取最晚，开始结束都没填写的时候取全部,格式为8位数字
    # # return dataframe 默认按照trade_date 从早到晚排序，最后一天在最下面
    # # ts_code trade_date open  high  low  close  pre_close change  pct_chg  vol  amount