# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:26:56 2019

@author: mgautama
"""

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns

import seaborn as sns


from datetime import datetime

import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima    
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor

import copy
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
from sklearn.multioutput import MultiOutputRegressor
import time
import datetime
import pickle
import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from datetime import date
from dateutil.rrule import rrule, MONTHLY
import pandas as pd
import pandas.io.sql
import pyodbc
import os
import calendar
import seaborn as sns
import numpy as np


from datetime import date
from dateutil.rrule import rrule, MONTHLY
import pandas as pd


from datetime import timedelta
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import sys


import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import ElasticNet


def get_first_day(dt, d_years=0, d_months=0):
    # d_years, d_months are "deltas" to apply to dt
    y, m = dt.year + d_years, dt.month + d_months
    a, m = divmod(m-1, 12)
    return date(y+a, m+1, 1)

def get_last_day(dt):
    return get_first_day(dt, 0, 1) + timedelta(-1)




def week_of_month(tgtdate):
	# tgtdate = tgtdate.to_datetime()

	days_this_month = calendar.mdays[tgtdate.month]
	for i in range(1,days_this_month):
		d = datetime.datetime(tgtdate.year, tgtdate.month, i)
		if d.day - d.weekday()>0:
			startdate = d
			break
	return (tgtdate-startdate).days//7+1


def feature_generation(data, date_col):
	
	df = data.copy()

	df['day'] = df[date_col].dt.day #day of month
	df['month'] = df[date_col].dt.month
	df['year'] = df[date_col].dt.year
	df['dayname'] = df[date_col].dt.day_name()
	df['weekend'] = [True if x in ['Saturday', 'Sunday'] else False for x in df['dayname']]
	
	#day of year
	df['day_of_year'] = df[date_col].dt.dayofyear
	#week of year
	df['week_number_y'] = df[date_col].dt.week
	#week of month
	df['week_of_month'] = df[date_col].apply(week_of_month)
	
	pub_holiday = pd.read_csv('public_holidays_2014_to_2020.csv', sep=';')
	pub_holiday['date'] = pd.to_datetime(pub_holiday['date'], format='%d/%m/%Y')
	df = pd.merge(df, pub_holiday, left_on=date_col, right_on='date', how='left')
	df['holiday'].fillna('Working Days', inplace=True)
	df.drop(columns=['date'], inplace=True)
	return df
    

def conditionsWeek(x):
    if x > 20:
        return 3
    elif x > 10:
        return 2
    else:
        return 1






#input any YYYY,mm,dd (it will try to validate the dd, if > max dd given month)
#then it will return day of last day of the month
#input as string for '05' will be still '05', since replacement only in
#date >10 
def normalizedate(var_year,var_mth,var_day):
    var_year_int = int(var_year)
    var_mth_int = int(var_mth)
    var_day_int = int(var_day)    
    
    var_date = date(var_year_int,var_mth_int,1)
    last_date = get_last_day(var_date)
    last_date_str = str(last_date)
    last_day = int(last_date_str[-2:])
    if var_day_int > last_day:
       return str(last_day)
    else:
       return str(var_day) 


def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
#    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
#    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        return "stationary"
    else:
        return "non-stationary"
    


def Startprog(startpar, endpar):

    
    
    funcnormalizedate = np.vectorize(normalizedate)
    funcWeek = np.vectorize(conditionsWeek)
    
    
    Data_rawx = pd.read_csv("OI_2016-2019_avg_3_6_v4.csv")
    Data_combine = Data_rawx[['Period']]
    Data_combine['Periods'] = Data_combine['Period']
    Data_combine.drop(columns = ['Period'], inplace=True)    
    Data_combine['NW (Ton)'] = Data_rawx['Monthly']
    Data_combine['lag_a'] = Data_rawx['Avg_3_Lag_3']  # at this code will not use the lag
    Data_combine['lag_b'] = Data_rawx['Avg_6_Lag_3']  # at this code will not use the lag
    Data_combine['B_ArtCode'] = Data_rawx['AC']
    Data_combine['B_Zone-3'] = Data_rawx['Zone']
 
#   REMOVE duplicate data, we only take National Data.
    Data_combine = Data_combine[Data_combine['B_Zone-3'] == 'National']
        
    
#    colnames=['Periods','NW (Ton)', 'RDD', 'B_ArtCode', 'B_Zone-3'] 
#    Data_raw1=pd.read_csv("OI_2016-2018.csv",usecols=colnames)
#    Data_raw2=pd.read_csv("OI_2019.csv",usecols=colnames)
    

#   Need to replace old article code with new article code
#   skip the header manually
    import csv
    reader = csv.reader(open('MappingNewArtCode.csv', 'r'))
    Data_map = {}
    ct_header = 0
    for row in reader:
       if ct_header == 0:
          ct_header = 1
       else:   
           k, v = row
           Data_map[int(k)] = int(v)

#   End Need to replace old article code with new article code


        
#    Data_combine = pd.concat([Data_raw1,Data_raw2],ignore_index=True)
    Data_combine = Data_combine.replace({"B_ArtCode": Data_map})
  
    
    
#   clean up data that has no article code, need to removed   
    Data_raw = pd.DataFrame()
    Data_raw = Data_combine[Data_combine.B_ArtCode.notnull()]
    

#   original data now has no 'days' because i copy paste from my prev code that require
#   date, so i pretend to assign date, for each month, using the last day of month
#   i just add date 31, then my previous Function has ability to fix the date if given
#   days in the date exceed the days in a month.


    
    Data_ori = pd.DataFrame()
    Data_ori['year'],Data_ori['month']  = Data_raw['Periods'].str.split(' ', 1).str
    #just give all month 31 days, then Function normilize data will fix based on the last day of month 
    Data_ori['day'] = 31    
    Data_ori['day'] = funcnormalizedate(Data_ori['year'],Data_ori['month'],Data_ori['day'])
    


    
    
    Data_ori['Date'] = Data_ori['year'] +'-'+ Data_ori['month'] + '-' + Data_ori['day']         
    Data_ori.drop(columns = ['year','month','day'], inplace=True)
    
    
    
#   end here generate the date, 

    Data_ori['Art_Code'] = Data_raw['B_ArtCode'].astype(str)
#    Data_ori['Art_Code'] = Data_ori['Art_Code'].apply(lambda x: x[:13])
    
    Data_ori['Zone'] =Data_raw['B_Zone-3']
    Data_ori['KEY'] = Data_ori['Art_Code'].astype(str)
    Data_ori['Quantity'] = Data_raw['NW (Ton)']
    Data_ori['lag_a'] = Data_raw['lag_a']
    Data_ori['lag_b'] = Data_raw['lag_b']
    

    
    
    ListArticle = Data_ori.groupby(['KEY'])['Quantity'].sum().sort_values(ascending=False).reset_index()
#   REMOVE LISTARTICLE THAT QTY TOTAL <= 0 (IT HAS 0 AND MINUS IN TOTAL)    
    ListArticle = ListArticle[ListArticle.Quantity > 0]    

    
    
    startindex = startpar
    endindex = endpar




    tmp_ListArticle = ListArticle[(ListArticle.index >= startindex) & (ListArticle.index <= endindex)]
    #tmp_ListArticle = ListArticle[(ListArticle.KEY == '1011001010020Zone 07')]
    
       
    ListArticle = tmp_ListArticle
    
    
    summaryresult = pd.DataFrame()
    excludeData = DataFrame()

    

    
    
    pred_start_dt = date(2018, 9, 1)
    pred_end_dt = date(2019, 9, 1)
    
    pred_range = rrule(MONTHLY, dtstart=pred_start_dt, until=pred_end_dt)
    
    train_start_dt = date(2018, 6, 1)
    train_end_dt = date(2019, 6, 1)
    
    train_range = rrule(MONTHLY, dtstart=train_start_dt, until=train_end_dt)
    
    
    for pred_dt,train_dt in zip (pred_range,train_range):
        pred_startmonth = date(pred_dt.year,pred_dt.month,pred_dt.day)
        pred_endmonth = get_last_day(pred_startmonth) 
        
        train_startmonth =  date(train_dt.year,train_dt.month,train_dt.day)
        train_endmonth = get_last_day(train_startmonth) 
        
        #change to string
        pred_startmonth = str(pred_startmonth)
        pred_endmonth = str(pred_endmonth)
        
        train_startmonth = str(train_startmonth)
        train_endmonth = str(train_endmonth)
        
#       print(train_endmonth,pred_startmonth,pred_endmonth )
    
    
    
    
        for iterArtCode in ListArticle.iterrows():
            
            keyArtCode = iterArtCode[1]['KEY']
        
        #    keyArtCode =  '1011001040020'
            
        
        
            Data_loop = Data_ori[Data_ori['KEY'] == keyArtCode ]
            del Data_loop['Zone']
            
        
            Data_loop[['Date']] = pd.to_datetime(Data_loop['Date'])                                    
            Data_loop = Data_loop.set_index('Date')
            Data_loop = Data_loop.resample('D').sum()       
            
            firstdate = Data_loop[0:1].index.get_values().astype('datetime64[D]')
            firstdate= firstdate[0].astype(str)
            r = pd.date_range(start=firstdate, end='2019-09-30')
        
                
            
            Data_loop = Data_loop.reindex(r).rename_axis('GI_Date')
            Data_loop = Data_loop.fillna(0)    
         
###########  new Set monthly data

            Data_monthly = copy.deepcopy(Data_loop)
            Data_monthly = Data_monthly.resample('M').sum()
 


############for all data
            all_data = copy.deepcopy(Data_monthly[(Data_monthly.index >= '2016-01-01') & (Data_monthly.index <= pred_endmonth)])
            all_data = all_data[8:]
           
            
############split train data
            train_based_data = copy.deepcopy(Data_monthly[(Data_monthly.index >= '2016-01-01') & (Data_monthly.index <= train_endmonth)])                        
#           need to cut train data so start data with lag_b not zero
            train_based_data = train_based_data[8:]
#           check for data not that long we dont' apply this logic 

            ln_train_based_data = len(train_based_data)    
            # as svr split 2 cv need at least 2 data
            if len(train_based_data) <=1:
                # need to LOG DATA
                cols = ['Concat','KEY','Month', 'QTYreal','QTYpredictSVRLinear','QTYpredictSVRRBF','QTYELASTIC','TXT_MODEL','CT_Train']
                lst = []
                txt_model ='DataNotEnough'
                lv_keyartcode = keyArtCode
                lv_concat = keyArtCode +'_'+ pred_endmonth
                lst.append([lv_concat,lv_keyartcode,pred_endmonth,Qtyreal,0,0,0,txt_model,ln_train_based_data])
                temprow = (pd.DataFrame(lst, columns=cols))
                summaryresult = summaryresult.append(temprow, ignore_index=True)  
               
                continue

             
############split test data           
            pred_based_data = copy.deepcopy(Data_monthly[(Data_monthly.index >= pred_startmonth) & (Data_monthly.index <= pred_endmonth)])
           
            
            
            
        #---- prepare x,y train    
            x_train = copy.deepcopy(train_based_data)
            x_train.drop(columns = ['Quantity'], inplace=True)
            
            y_train = train_based_data['Quantity']
        #---- prepare x,y train       
        #---- prepare x,y test 
            x_test = copy.deepcopy(pred_based_data)
            x_test.drop(columns = ['Quantity'], inplace=True)
            
            y_test = pred_based_data['Quantity']
        
        #---- prepare x,y test 
        
        #---- prepare x,y all
            x_all_data = copy.deepcopy(all_data)
            x_all_data.drop(columns = ['Quantity'], inplace=True)
            
            y_all_data = all_data['Quantity']
        
        #---- prepare x,y all   
        
                   

            lv_cv = 2
        
            svr = GridSearchCV(SVR(kernel='linear'), cv=lv_cv,
                               param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                           "gamma": np.logspace(-2, 2, 5)})    
        
            
            svr.fit(x_train, y_train)
                        
            
            y_pred = svr.predict(x_test)
        
            summary_pred_svr = y_pred.sum()
            Qtyreal = y_test.sum()
            
            
#            y_pred_all_svr = svr.predict(x_all_data)
#            plt.figure(figsize=(10, 10))
            
#            y_true = pd.DataFrame(y_all_data).reset_index()
            
            
#            plt.figure(figsize=(10, 10))
#            plt.plot(y_true['Quantity'] )
#            plt.plot(y_pred_all_svr,'r')             
 


            lv_cv = 2
        
            svr = GridSearchCV(SVR(kernel='rbf'), cv=lv_cv,
                               param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                           "gamma": np.logspace(-2, 2, 5)})    
        
            
            svr.fit(x_train, y_train)
                        
            
            y_pred = svr.predict(x_test)
        
            summary_pred_svr_rbf = y_pred.sum()
            Qtyreal = y_test.sum()

           
            
            
                # END else use SVR
        #---> End predict normal   
        


############       elastic net
            parametersGrid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1],"l1_ratio": np.arange(0.0, 1.0, 0.1)}

            eNet = ElasticNet()
            
            grid = GridSearchCV(eNet, parametersGrid, scoring='r2', cv=2)
            model = grid.fit(x_train, y_train)
            best_params = model.best_params_
            y_pred_enet = model.predict(x_test)           
                
            summary_pred_enet = y_pred_enet.sum()
            Qtyreal = y_test.sum()        
            
 

#            y_pred_all_enet = model.predict(x_all_data)
            
#            y_true = pd.DataFrame(y_all_data).reset_index()
            
            
#            plt.figure(figsize=(10, 10))
#            plt.plot(y_true['Quantity'] )
#            plt.plot(y_pred_all_enet,'r')               
            
            
############       elastic net            
        
        
            #suppress the result prediction if the last 3 month data avg is zero
            txt_model =''
            checknull = x_test['lag_a'][0]
            if checknull <= 0:
               summary_pred_svr = 0
               summary_pred_svr_rbf = 0
               summary_pred_enet = 0
               txt_model ='Set_to_ZERO'

                
               
            
            
     
            cols = ['Concat','KEY','Month', 'QTYreal','QTYpredictSVRLinear','QTYpredictSVRRBF','QTYELASTIC','TXT_MODEL','CT_Train']
            lst = []
            lv_keyartcode = keyArtCode
            lv_concat = keyArtCode +'_'+ pred_endmonth
            lst.append([lv_concat,lv_keyartcode,pred_endmonth,Qtyreal,summary_pred_svr,summary_pred_svr_rbf,summary_pred_enet,txt_model,ln_train_based_data])
            temprow = (pd.DataFrame(lst, columns=cols))
            summaryresult = summaryresult.append(temprow, ignore_index=True)   
         
    
    #---> end predict with scaler
    
    
    filename = './Result/Art_code_level_WITH_LAG_ARIMA_SVR_ELASTIC_OI_Monthly_' + str(startindex)+'_'+str(endindex)
    summaryresult.to_csv(filename+'.csv')

if __name__ == '__main__':
    Startprog(int(sys.argv[1]), int(sys.argv[2]))    