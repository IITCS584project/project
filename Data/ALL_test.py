 #!/usr/bin/python
# -*- coding: UTF-8 -*-


import time
import datetime
import sys
import csv
import json
import pandas as pd
import os
import sys
sys.path.append("../")
from InvestmentAnalystSystem.BayesianMethodAnalyst.GaussianNaiveBayesMethod import Main as Bayes_main
from InvestmentAnalystSystem.NNPricingSystem.MultiFeatureNNPricingSystem import Main as NN_main
from InvestmentAnalystSystem.LSTM.KerasLSTM import stock_NN_v1 as stock_NN_v1
from InvestmentAnalystSystem.LinearAnalyst.MultiFeatureSystem import Main as Multi_main
from InvestmentAnalystSystem.LinearAnalyst.CAPM import Main  as CAPM_main
from Data.UseData import read_data


obj_read=read_data()
list_=obj_read.get_trade_cal_stock_list(start_date=20190101,end_date=20210201)
list_=list_[0:1000]
#print(len(list_))
#list_=['000001.SZ','000002.SZ']
train_start=20190101
train_end=20200101
test_start=20200101
test_end=20210201

def test_Bayes(train_start=train_start,train_end=train_end,test_start=test_start,test_end=test_end):
	result={}
	i=0
	for code in list_:
	    i+=1
	    try:
	    	tmp=Bayes_main(code=code,train_start=train_start,train_end=train_end,test_start=test_start,test_end=test_end)
	    	result[code]=tmp
	    except Exception as  e:
	    	print(str(e))
	    print(code,tmp,i,len(list_))
	return result
def test_NN(train_start=train_start,train_end=train_end,test_start=test_start,test_end=test_end):
	result={}
	i=0
	for code in list_:
		i+=1
		try:
			tmp=NN_main(code=code,train_start=train_start,train_end=train_end,test_start=test_start,test_end=test_end)
			result[code]=tmp
		except Exception as  e:
			print(str(e))
		print(code,tmp,i,len(list_))
	return result

def test_LSTM(train_start=train_start,train_end=train_end,test_start=test_start,test_end=test_end):
	result={}
	i=0
	for code in list_:
		i+=1
		try:
			tmp=stock_NN_v1(code=code,train_start=train_start,train_end=train_end,test_start=test_start,test_end=test_end)
			result[code]=tmp
		except Exception as  e:
			print(str(e))
		print(code,tmp,i,len(list_))
	return result


def test_CAPM(train_start=train_start,train_end=train_end,test_start=test_start,test_end=test_end):
	result={}
	i=0
	tmp=''
	for code in list_:
		i+=1
		try:
			tmp=CAPM_main(code=code,train_start=train_start,train_end=train_end,test_start=test_start,test_end=test_end)
			result[code]=tmp
		except Exception as  e:
			print(str(e))
		print(code,tmp,i,len(list_))
	return result


def test_MF(train_start=train_start,train_end=train_end,test_start=test_start,test_end=test_end):
	result={}
	i=0
	for code in list_:
		i+=1
		try:
			tmp=Multi_main(code=code,train_start=train_start,train_end=train_end,test_start=test_start,test_end=test_end)
			result[code]=tmp
		except Exception as  e:
			print(str(e))
		print(code,tmp,i,len(list_))
	return result


def write_csv(name,res):

	f = open(name+'.csv','w')
	for key in res.keys():
		
		f.write(key+'   '+str(res.get(key))+'\n')
	f.close()


if __name__ == '__main__':
	print(3)
	res=test_MF()
	write_csv('test_MF',res)
	#res=test_CAPM()
	#write_csv('test_CAPM',res)
