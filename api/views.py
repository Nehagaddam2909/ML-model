#import response from rest framework
from rest_framework.response import Response
from rest_framework.decorators import api_view
from datetime import datetime
import csv
import numpy as np
import pandas as pd
from prophet import Prophet

#import file system
# import FileSystemStorage
from django.core.files.storage import FileSystemStorage
fs=FileSystemStorage(location='data/')
import json
@api_view(['POST'])  
def getData(request):
    # print(request.body)
    # d=request.body.import traceback; traceback.print_exc();
    #convert reuest body to json
    data=json.loads(request.body)
    print(data)
    train = pd.read_csv('https://raw.githubusercontent.com/imdevskp/covid-19-india-data/master/complete.csv')
    train.rename(columns = {"Total Confirmed cases":'Confirmed',"Name of State / UT":"State","Cured/Discharged/Migrated":"Recovered"}, inplace = True)
    train.State = train.State.fillna('Not Available')
    train.sort_values('Date').reset_index().drop('index', axis =1)
    train.groupby('Date')[['Confirmed','State','Death']].sum().reset_index()
    train_west = train.loc[train['State']==data['state']].copy()
    train_west['Date'] = pd.to_datetime(train_west['Date'])
    train_west = train_west.set_index('Date')
    train_ind2 = train_west.loc[train_west['Confirmed']>0]
    
    new_colname = 'y'
    train_ind2.rename(columns = {'Confirmed' : 'y'},inplace=True)
    train_ind2.reset_index(level=0, inplace=True)
    
    model = Prophet(
    interval_width=0.95,
    holidays = pd.DataFrame({'holiday': 'lockdown','ds': pd.to_datetime(['2020-03-24','2020-03-25','2020-03-26','2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31','2020-04-01'
    ,'2020-04-02','2020-04-03','2020-04-04','2020-04-05','2020-04-05','2020-04-06','2020-04-07','2020-04-08','2020-04-09','2020-04-10','2020-04-11','2020-04-12','2020-04-13','2020-04-14'])}),
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )
    train_ind2.rename(columns = {'Date' : 'ds'},inplace=True)
    print("model:",train_ind2)
    # train_ind2.
    model.fit(train_ind2)
    future_pd = model.make_future_dataframe(
    periods=60,
    freq='d',
    include_history=True
    )
    
    forecast_pd = model.predict(future_pd)
    forecast_pd[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[55:63]
    fpd = pd.DataFrame(forecast_pd[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    fpd.rename(columns= {'ds':'Date', 'yhat':'Predicted Cases of Kerala'},inplace=True)

    fpd.drop(['yhat_lower','yhat_upper'],axis=1,inplace=True)

    #prediction of cases in Westbengal for date 28/04/2020
    k=fpd[(fpd['Date']>= '2020-04-28') & (fpd['Date']<= '2020-04-28')]
    print("k:",k)

    # type of k 
    print("type of k:",type(k))


    person={'name':'John','age':23}
    print("This works")
    k=k.to_json(orient='records')
    print("This not")
    try:
        # convert k to string and then to json
        parsed_data = json.loads(k) 
        print("parsed_data:",parsed_data)
        # Convert Timestamp to a JSON-serializable format
        parsed_data[0]["Date"] = datetime.fromtimestamp(parsed_data[0]["Date"] / 1000).isoformat()
        print("parsed_data after process:",parsed_data)
        # prettified_data = json.dumps(parsed_data, indent=4)
        # print("prettified_data:",prettified_data)
        return Response(parsed_data[0], content_type='application/json')
    except json.JSONDecodeError:
        return Response("Invalid data")

    # return Response(json.dumps(k.to_dict(orient='records')))

